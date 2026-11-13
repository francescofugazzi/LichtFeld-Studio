/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "app/include/app/converter.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
#include "io/project_document.hpp"
#include "licht_test_support.hpp"
#include "training/checkpoint.hpp"
#include "training/strategies/mcmc.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::core::param;
using namespace lfs::io;
using namespace lfs::io::project;
using lfs::test::licht::fingerprint;
using lfs::test::licht::fixed_uuid_in_namespace;
using lfs::test::licht::make_empty_document;
using lfs::test::licht::make_splat;
using lfs::test::licht::require_status;

namespace {

    constexpr std::uint32_t kUuidNamespace = 0x7c000000;

    class ScopedInputBuffer {
    public:
        explicit ScopedInputBuffer(std::streambuf* replacement)
            : previous_(std::cin.rdbuf(replacement)) {}

        ~ScopedInputBuffer() { std::cin.rdbuf(previous_); }

        ScopedInputBuffer(const ScopedInputBuffer&) = delete;
        ScopedInputBuffer& operator=(const ScopedInputBuffer&) = delete;

    private:
        std::streambuf* previous_;
    };

    Uuid test_uuid(const std::uint64_t tag) {
        return fixed_uuid_in_namespace(kUuidNamespace, tag);
    }

    ProjectDocumentSaveOptions save_options(
        const std::uint64_t identity_tag, const std::uint64_t wallclock) {
        return lfs::test::licht::deterministic_document_save_options(
            kUuidNamespace, identity_tag, wallclock);
    }

    std::unique_ptr<SplatData> make_shaped_splat(
        const std::size_t count, const int sh_degree, const float first_opacity = 0.0f) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        std::vector<float> opacity(count, 0.0f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index);
            rotations[index * 4] = 1.0f;
        }
        opacity.front() = first_opacity;
        const auto rest = static_cast<std::size_t>((sh_degree + 1) * (sh_degree + 1) - 1);
        return std::make_unique<SplatData>(
            sh_degree, Tensor::from_vector(means, {count, std::size_t{3}}, Device::CPU),
            Tensor::zeros({count, std::size_t{1}, std::size_t{3}}, Device::CPU, DataType::Float32),
            rest ? Tensor::ones({count, rest, std::size_t{3}}, Device::CPU, DataType::Float32) : Tensor{},
            Tensor::zeros({count, std::size_t{3}}, Device::CPU, DataType::Float32),
            Tensor::from_vector(rotations, {count, std::size_t{4}}, Device::CPU),
            Tensor::from_vector(opacity, {count, std::size_t{1}}, Device::CPU), 1.0f);
    }

    std::array<float, 16> translation(const float x, const float y, const float z) {
        return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1};
    }

    std::shared_ptr<SplatData> load_output(const fs::path& path) {
        auto loaded = Loader::create()->load(path);
        EXPECT_TRUE(loaded) << (loaded ? std::string{} : loaded.error().format());
        if (!loaded)
            return nullptr;
        auto* splat = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
        return splat ? *splat : nullptr;
    }

    std::vector<float> sorted_column(const SplatData& splat, const int column) {
        const auto values = splat.means().to(Device::CPU).contiguous().to_vector();
        std::vector<float> result;
        for (std::size_t row = 0; row < values.size() / 3; ++row)
            result.push_back(values[row * 3 + static_cast<std::size_t>(column)]);
        std::ranges::sort(result);
        return result;
    }

    ConvertParameters make_convert_params(const fs::path& input, const fs::path& output) {
        ConvertParameters params;
        params.input_path = input;
        params.output_path = output;
        params.format = OutputFormat::PLY;
        params.sh_degree = -1;
        params.overwrite = true;
        return params;
    }

    class ConverterLichtTest : public ::testing::Test {
    protected:
        const fs::path temp_dir = fs::temp_directory_path() / "lfs_converter_licht_test";

        void SetUp() override {
            fs::remove_all(temp_dir);
            fs::create_directories(temp_dir);
        }

        void TearDown() override {
            fs::remove_all(temp_dir);
        }

        void add_root_group(ProjectDocument& document, const Uuid& root_uuid) {
            require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = root_uuid,
                .type = "group",
                .name = "Root",
                .child_order = 0,
            }));
        }

        void add_embedded_splat(
            ProjectDocument& document,
            const Uuid& node_uuid,
            const Uuid& parent_uuid,
            const std::uint32_t child_order,
            const SplatData& model,
            const std::string& name,
            const std::uint8_t fingerprint_tag,
            const std::array<float, 16>& local_transform = translation(0, 0, 0),
            const bool visible = true) {
            require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = node_uuid,
                .type = "splat",
                .name = name,
                .parent_uuid = parent_uuid,
                .child_order = child_order,
                .local_transform = local_transform,
                .visible = visible,
                .payload =
                    PayloadBinding{
                        .fourcc = "SPLT",
                        .instance_uuid = node_uuid,
                        .source_kind = "ply",
                    },
            }));
            auto captured = SplatChapterPayload::capture(
                model, SplatSourceKind::ImportedPly, false);
            ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
            require_status(document.set_splat(node_uuid, std::move(*captured)));

            auto& project = document.edit_project();
            require_status(project.upsert_embed_decision(EmbedDecision{
                .uuid = node_uuid,
                .node_uuid = node_uuid,
                .payload_fourcc = "SPLT",
                .decision = "embedded",
                .reason = "convert licht test",
            }));
            require_status(project.upsert_embedded_payload_provenance(
                EmbeddedPayloadProvenance{
                    .uuid = node_uuid,
                    .node_uuid = node_uuid,
                    .fourcc = "SPLT",
                    .import_locator =
                        {
                            .preferred = "assets/splat.ply",
                            .base = LocatorBase::Project,
                        },
                    .import_fingerprint = fingerprint(fingerprint_tag),
                    .content_xxh3_128 = {},
                }));
        }

        void add_group(ProjectDocument& document, const Uuid& uuid, const Uuid& parent,
                       const std::uint32_t child_order, const std::array<float, 16>& transform,
                       const bool visible) {
            require_status(document.edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = uuid,
                .type = "group",
                .name = "Group",
                .parent_uuid = parent,
                .child_order = child_order,
                .local_transform = transform,
                .visible = visible,
            }));
        }

        void write_published_project(const fs::path& path, const JsonChapterDom::Json& publication) {
            const auto source = temp_dir / "published-source.ply";
            ASSERT_TRUE(save_ply(*make_shaped_splat(3, 1), {.output_path = source, .binary = true}));
            auto document = make_empty_document(test_uuid(300), 100);
            const auto id = test_uuid(301);
            require_status(document->edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = id,
                .type = "splat",
                .name = "Published",
                .payload = PayloadBinding{.fourcc = "DSRC", .instance_uuid = id, .source_kind = "ply"},
            }));
            auto node = document->edit_scene_graph().dom().array_find("nodes", id.to_string());
            ASSERT_TRUE(node);
            require_status(node->set_json("publication", publication));
            auto payload = LazyChunkValue::from_owned(lfs::test::licht::read_file_bytes(source), id);
            ASSERT_TRUE(payload) << lfs::format_for_developer(payload.error());
            require_status(document->set_dataset_source(id, std::move(*payload)));
            const auto saved = document->save(path, save_options(310, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_empty_project(const fs::path& path) {
            auto document = make_empty_document(test_uuid(1), 100);
            const auto saved = document->save(path, save_options(10, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_splt_project(
            const fs::path& path, const SplatData& model, const std::uint64_t identity_tag) {
            auto document = make_empty_document(test_uuid(identity_tag), 100);
            const auto root_uuid = test_uuid(identity_tag + 1);
            const auto splat_uuid = test_uuid(identity_tag + 2);
            add_root_group(*document, root_uuid);
            add_embedded_splat(
                *document, splat_uuid, root_uuid, 0, model, "Splat",
                static_cast<std::uint8_t>(identity_tag & 0xff));
            const auto saved = document->save(path, save_options(identity_tag + 10, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_two_splt_project(const fs::path& path) {
            auto document = make_empty_document(test_uuid(80), 100);
            const auto root_uuid = test_uuid(81);
            add_root_group(*document, root_uuid);
            add_embedded_splat(*document, test_uuid(82), root_uuid, 0, *make_splat(2), "A", 2);
            add_embedded_splat(*document, test_uuid(83), root_uuid, 1, *make_splat(3), "B", 3);
            const auto saved = document->save(path, save_options(90, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        void write_ckpt_project(const fs::path& path, const int gaussian_count) {
            auto model = make_splat(static_cast<std::size_t>(gaussian_count));
            lfs::training::MCMC strategy(*model);
            TrainingParameters parameters;
            parameters.optimization = OptimizationParameters::mcmc_defaults();
            parameters.optimization.sh_degree = 0;
            parameters.optimization.max_cap = gaussian_count;

            std::ostringstream stream(std::ios::binary | std::ios::out);
            const auto serialized = lfs::training::serialize_checkpoint(
                stream, 17, strategy, parameters, nullptr, nullptr, nullptr);
            ASSERT_TRUE(serialized) << lfs::format_for_developer(serialized.error());
            const auto encoded = stream.str();
            ASSERT_FALSE(encoded.empty());
            std::vector<std::byte> bytes(encoded.size());
            std::memcpy(bytes.data(), encoded.data(), encoded.size());

            const auto project_uuid = test_uuid(200);
            const auto root_uuid = test_uuid(201);
            const auto training_uuid = test_uuid(202);
            const auto checkpoint_uuid = test_uuid(203);
            auto document = make_empty_document(project_uuid, 100);
            add_root_group(*document, root_uuid);
            require_status(document->edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = training_uuid,
                .type = "splat",
                .name = "Training",
                .parent_uuid = root_uuid,
                .child_order = 0,
                .payload =
                    PayloadBinding{
                        .fourcc = "CKPT",
                        .instance_uuid = checkpoint_uuid,
                        .source_kind = "training",
                    },
            }));
            require_status(document->edit_scene_graph().set_training_model_uuid(training_uuid));
            auto payload = LazyChunkValue::from_owned(std::move(bytes), checkpoint_uuid);
            ASSERT_TRUE(payload) << lfs::format_for_developer(payload.error());
            require_status(document->set_checkpoint(checkpoint_uuid, std::move(*payload)));

            auto options = save_options(210, 200);
            options.commit.snapshot_uuid = checkpoint_uuid;
            const auto saved = document->save(path, options);
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }
    };

    TEST_F(ConverterLichtTest, ConvertsSingleSpltProjectToPly) {
        const auto project = temp_dir / "single-splt.licht";
        const auto output = temp_dir / "single-splt.ply";
        constexpr std::size_t kCount = 4;
        write_splt_project(project, *make_splat(kCount), 20);

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        (void)testing::internal::GetCapturedStdout();
        ASSERT_EQ(rc, 0);
        ASSERT_TRUE(fs::exists(output));

        auto loaded = load_ply(output);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->value.size(), kCount);
        EXPECT_EQ(loaded->value.get_max_sh_degree(), 0);
    }

    TEST_F(ConverterLichtTest, ConvertsSingleCkptProjectToPly) {
        const auto project = temp_dir / "single-ckpt.licht";
        const auto output = temp_dir / "single-ckpt.ply";
        constexpr int kCount = 3;
        write_ckpt_project(project, kCount);

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        (void)testing::internal::GetCapturedStdout();
        ASSERT_EQ(rc, 0);
        ASSERT_TRUE(fs::exists(output));

        auto loaded = load_ply(output);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->value.size(), static_cast<std::size_t>(kCount));
        EXPECT_EQ(loaded->value.get_max_sh_degree(), 0);
    }

    TEST_F(ConverterLichtTest, RejectsProjectWithNoSplatModel) {
        const auto project = temp_dir / "empty.licht";
        const auto output = temp_dir / "empty.ply";
        write_empty_project(project);

        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        const auto stdout_text = testing::internal::GetCapturedStdout();
        const auto stderr_text = testing::internal::GetCapturedStderr();

        EXPECT_NE(rc, 0);
        EXPECT_NE(stderr_text.find("no visible splats"), std::string::npos);
        EXPECT_EQ(stdout_text.find("Loaded"), std::string::npos);
        EXPECT_FALSE(fs::exists(output));
    }

    TEST_F(ConverterLichtTest, ConvertsEveryModelOfMultiModelProjectWithOptions) {
        const auto project = temp_dir / "two-models.licht";
        const auto output = temp_dir / "two-models.spz";
        write_two_splt_project(project);
        auto params = make_convert_params(project, output);
        params.format = OutputFormat::SPZ;
        params.spz_version = 3;

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(params);
        const auto stdout_text = testing::internal::GetCapturedStdout();

        ASSERT_EQ(rc, 0);
        EXPECT_NE(stdout_text.find("Merged 2 visible splat nodes"), std::string::npos);
        const auto loaded = load_output(output);
        ASSERT_TRUE(loaded);
        EXPECT_EQ(loaded->size(), 5u);
    }

    TEST_F(ConverterLichtTest, AppliesSceneVisibilityAndTransformsAndReportsCrops) {
        const auto project = temp_dir / "scene.licht";
        const auto output = temp_dir / "scene.ply";
        {
            auto document = make_empty_document(test_uuid(400), 100);
            const auto root = test_uuid(401);
            const auto moved = test_uuid(402);
            const auto hidden = test_uuid(403);
            const auto moved_splat = test_uuid(404);
            add_root_group(*document, root);
            add_group(*document, moved, root, 0, translation(10, 0, 0), true);
            add_group(*document, hidden, root, 1, translation(0, 0, 0), false);
            add_embedded_splat(*document, moved_splat, moved, 0, *make_splat(2), "Moved", 4,
                               translation(0, 0, 3));
            add_embedded_splat(*document, test_uuid(405), root, 2, *make_splat(3), "Raised", 5,
                               translation(0, 5, 0));
            add_embedded_splat(*document, test_uuid(406), hidden, 0, *make_splat(4), "Under hidden", 6);
            add_embedded_splat(*document, test_uuid(407), root, 3, *make_splat(6), "Hidden", 7,
                               translation(0, 0, 0), false);
            require_status(document->edit_scene_graph().upsert_node(SceneNodeRecord{
                .uuid = test_uuid(408),
                .type = "cropbox",
                .name = "Crop",
                .parent_uuid = moved_splat,
                .cropbox = CropBoxRecord{.enabled = true},
            }));
            const auto saved = document->save(project, save_options(410, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        (void)testing::internal::GetCapturedStdout();
        const auto stderr_text = testing::internal::GetCapturedStderr();

        ASSERT_EQ(rc, 0);
        EXPECT_NE(stderr_text.find("1 crop volume(s)"), std::string::npos);
        const auto loaded = load_output(output);
        ASSERT_TRUE(loaded);
        ASSERT_EQ(loaded->size(), 5u);
        EXPECT_EQ(sorted_column(*loaded, 0), (std::vector<float>{0, 1, 2, 10, 11}));
        EXPECT_EQ(sorted_column(*loaded, 1), (std::vector<float>{0, 0, 5, 5, 5}));
        EXPECT_EQ(sorted_column(*loaded, 2), (std::vector<float>{0, 0, 0, 3, 3}));
    }

    TEST_F(ConverterLichtTest, RemovesNoiseInfillFromEachSourceBeforeMerging) {
        const auto project = temp_dir / "noise.licht";
        const auto output = temp_dir / "noise.ply";
        auto noisy = make_shaped_splat(4, 0, kHiddenOpacity);
        noisy->set_frozen_ranges({{.start = 0, .count = 2}});
        ASSERT_EQ(noise_infill_rows(*noisy), 2u);
        {
            auto document = make_empty_document(test_uuid(500), 100);
            const auto root = test_uuid(501);
            add_root_group(*document, root);
            add_embedded_splat(*document, test_uuid(502), root, 0, *make_splat(3), "Clean", 8);
            add_embedded_splat(*document, test_uuid(503), root, 1, *noisy, "Noisy", 9);
            const auto saved = document->save(project, save_options(510, 200));
            ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        }

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(project, output));
        (void)testing::internal::GetCapturedStdout();

        ASSERT_EQ(rc, 0);
        const auto loaded = load_output(output);
        ASSERT_TRUE(loaded);
        EXPECT_EQ(loaded->size(), 5u);
        EXPECT_EQ(sorted_column(*loaded, 0), (std::vector<float>{0, 1, 2, 2, 3}));
    }

    TEST_F(ConverterLichtTest, UsesPublishedShDegreeAndRejectsInvalidPublication) {
        const auto published = temp_dir / "published.licht";
        const auto output = temp_dir / "published.ply";
        using Json = JsonChapterDom::Json;
        write_published_project(published, Json{{"count", 3}, {"sh_degree", 0}});

        testing::internal::CaptureStdout();
        ASSERT_EQ(lfs::app::run_converter(make_convert_params(published, output)), 0);
        (void)testing::internal::GetCapturedStdout();
        const auto loaded = load_output(output);
        ASSERT_TRUE(loaded);
        EXPECT_EQ(loaded->size(), 3u);
        EXPECT_EQ(loaded->get_max_sh_degree(), 0);

        for (const auto& [name, publication] : {
                 std::pair{"text-degree", Json{{"count", 3}, {"sh_degree", "1"}}},
                 std::pair{"wrong-count", Json{{"count", 4}, {"sh_degree", 0}}},
             }) {
            const auto invalid = temp_dir / (std::string(name) + ".licht");
            const auto rejected = temp_dir / (std::string(name) + ".ply");
            write_published_project(invalid, publication);
            testing::internal::CaptureStdout();
            testing::internal::CaptureStderr();
            const int rc = lfs::app::run_converter(make_convert_params(invalid, rejected));
            (void)testing::internal::GetCapturedStdout();
            const auto stderr_text = testing::internal::GetCapturedStderr();
            EXPECT_NE(rc, 0) << name;
            EXPECT_NE(stderr_text.find("published splat metadata"), std::string::npos) << name;
            EXPECT_FALSE(fs::exists(rejected)) << name;
        }
    }

    TEST_F(ConverterLichtTest, DirectoryScanFindsLichtProjects) {
        const auto input_dir = temp_dir / "input";
        const auto output_dir = temp_dir / "output";
        fs::create_directories(input_dir);
        fs::create_directories(output_dir);
        constexpr std::size_t kCount = 2;
        write_splt_project(input_dir / "scene.licht", *make_splat(kCount), 40);

        testing::internal::CaptureStdout();
        const int rc = lfs::app::run_converter(make_convert_params(input_dir, output_dir));
        (void)testing::internal::GetCapturedStdout();
        ASSERT_EQ(rc, 0);

        const auto output = output_dir / "scene_converted.ply";
        ASSERT_TRUE(fs::exists(output));
        auto loaded = load_ply(output);
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->value.size(), kCount);
    }

    TEST_F(ConverterLichtTest, OverwritePromptPreservesUnicodeFilename) {
        const auto project = temp_dir / "input.licht";
        const auto output = temp_dir / lfs::core::utf8_to_path("出力_é.ply");
        write_empty_project(project);
        {
            std::ofstream existing(output, std::ios::binary);
            ASSERT_TRUE(existing.is_open());
        }

        auto params = make_convert_params(project, output);
        params.overwrite = false;
        int rc = 1;
        std::string stdout_text;
        {
            std::istringstream input("n\n");
            const ScopedInputBuffer redirect_input(input.rdbuf());
            testing::internal::CaptureStdout();
            rc = lfs::app::run_converter(params);
            stdout_text = testing::internal::GetCapturedStdout();
        }

        EXPECT_EQ(rc, 0);
        EXPECT_NE(stdout_text.find(lfs::core::path_to_utf8(output.filename())),
                  std::string::npos);
        EXPECT_NE(stdout_text.find("1 skipped"), std::string::npos);
    }

} // namespace
