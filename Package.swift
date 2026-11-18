// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "QwenFieldfare",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "FlintFieldfareCore", targets: ["FlintFieldfareCore"]),
        .executable(name: "QwenFieldfareServer", targets: ["QwenFieldfareServer"]),
        .library(name: "QwenFieldfareCore", targets: ["QwenFieldfareCore"]),
        .executable(name: "QwenFieldfareMac", targets: ["QwenFieldfareMac"]),
        .executable(name: "QwenFieldfareBench", targets: ["QwenFieldfareBench"]),
        .executable(name: "QwenFieldfarePack", targets: ["QwenFieldfarePack"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.3.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(name: "QwenFieldfareServerManagement", dependencies: ["QwenFieldfareCore", "QwenFieldfareServerSupport"]),
        .testTarget(name: "QwenFieldfareServerManagementTests", dependencies: ["QwenFieldfareServerManagement", "QwenFieldfareServerSupport"]),
        .target(name: "QwenFieldfareServerSupport", dependencies: ["QwenFieldfareCore", .product(name: "Jinja", package: "swift-jinja"), .product(name: "NIOCore", package: "swift-nio"), .product(name: "NIOHTTP1", package: "swift-nio"), .product(name: "NIOPosix", package: "swift-nio")]),
        .executableTarget(name: "QwenFieldfareServer", dependencies: ["QwenFieldfareServerSupport", "QwenFieldfareCore"]),
        .testTarget(name: "QwenFieldfareServerTests", dependencies: ["QwenFieldfareServerSupport"], resources: [.copy("Fixtures")]),
        .target(
            name: "QwenFieldfareCore",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            path: "Sources/QwenFieldfareCore",
            sources: [
                "Infrastructure/Metal/MetalContext.swift",
                "Infrastructure/ModelIO/QwenModelDirectory.swift",
                "Infrastructure/ModelIO/QwenInferenceLease.swift",
                "Infrastructure/ModelIO/QwenResidentCoreLoader.swift",
                "Infrastructure/ModelIO/QwenSafetensorsHeader.swift",
                "Infrastructure/ModelIO/QwenMappedSafetensorsFile.swift",
                "Infrastructure/ModelIO/Sha256Verifier.swift",
                "Kernels/Attention/Attention.swift",
                "Kernels/LinearAttention/QwenDeltaNetKernel.swift",
                "Runtime/Generation/QwenActiveStreamingMemoryConfiguration.swift",
                "Runtime/Generation/QwenGenerationSupport.swift",
                "Runtime/Generation/QwenQ4ActiveExpertStreamingRunner.swift",
                "Runtime/Generation/QwenQ4ChatRunner.swift",
                "Runtime/Inference/QwenForwardRunner.swift",
                "Runtime/Telemetry/QwenRuntimeTelemetry.swift",
                "Storage/Q4BoundedExpertCache.swift",
                "Storage/Q4ExpertPackedStore.swift",
                "Storage/QwenQ4ExpertStore.swift",
                "Storage/QwenQ4ExpertStoreBuilder.swift",
            ],
            resources: [.copy("Metal")]
        ),
        .target(
            name: "QwenFieldfarePresentation",
            path: "Sources/QwenFieldfareApp/MacPresentation",
            sources: [
                "AboutPanelPresentation.swift",
                "FastMarkdownParser.swift",
                "ModelRevealPolicy.swift",
            ]
        ),
        .executableTarget(
            name: "QwenFieldfareMac",
            dependencies: ["QwenFieldfareCore", "QwenFieldfarePresentation", "QwenFieldfareServerManagement"],
            path: "Sources/QwenFieldfareApp/Mac",
            sources: [
                "Server/ServerSettingsView.swift",
                "App/MacAppIcon.swift",
                "App/RootView.swift",
                "App/QwenFieldfareMacApp.swift",
                "Diagnostics/ChatInspectorView.swift",
                "Generation/ChatMessage.swift",
                "Generation/ChatMessageView.swift",
                "Generation/ChatPromptInputView.swift",
                "Generation/ChatScrollView.swift",
                "Generation/ChatService.swift",
                "Generation/CodeBlockView.swift",
                "Generation/MarkdownContentView.swift",
            ],
            resources: [.copy("Resources/qwenfieldfare-app-icon.png")]
        ),
        .executableTarget(
            name: "QwenFieldfareBench",
            dependencies: ["QwenFieldfareCore"],
            path: "Sources/QwenFieldfareBench"
        ),
        .executableTarget(
            name: "QwenFieldfarePack",
            dependencies: ["QwenFieldfareCore"],
            path: "Sources/QwenFieldfarePack"
        ),
        .testTarget(
            name: "QwenFieldfareCoreTests",
            dependencies: ["QwenFieldfareCore"],
            path: "Tests/QwenFieldfareCore",
            sources: [
                "Kernels/Attention/QwenFP16AttentionBridgeTests.swift",
                "QwenDeltaNetMathTests.swift",
                "QwenActiveStreamingMemoryConfigurationTests.swift",
                "QwenExpertReductionTests.swift",
                "QwenGenerationSupportTests.swift",
                "QwenModelDirectoryTests.swift",
                "QwenInferenceLeaseTests.swift",
                "QwenSafetensorsHeaderTests.swift",
                "QwenResidentWeightsTests.swift",
                "QwenMappedSafetensorsTests.swift",
                "QwenFallbackReaderTests.swift",
                "QwenRuntimeTelemetryTests.swift",
                "Storage/Q4BoundedExpertCacheTests.swift",
                "Storage/QwenQ4ExpertStoreTests.swift",
            ]
        ),
        .testTarget(
            name: "QwenFieldfarePresentationTests",
            dependencies: ["QwenFieldfarePresentation"],
            path: "Tests/QwenFieldfarePresentation",
            sources: [
                "AboutPanelPresentationTests.swift",
                "FastMarkdownParserTests.swift",
                "ModelRevealPolicyTests.swift",
            ]
        ),
        .target(
            name: "FlintFieldfareCore",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            path: "Sources/FlintFieldfareCore",
            sources: [
                "Infrastructure/Metal/FlintMetalContext.swift",
                "Infrastructure/ModelIO/FlintInferenceLease.swift",
                "Infrastructure/ModelIO/FlintModelDirectory.swift",
                "Infrastructure/ModelIO/FlintSafetensorsFile.swift",
                "Infrastructure/ModelIO/FlintResidentBuffer.swift",
                "Infrastructure/ModelIO/FlintResidentLoader.swift",
                "Infrastructure/ModelIO/FlintResidentManifest.swift",
                "Infrastructure/ModelIO/FlintResidentWeights.swift",
                "Kernels/FlintRMSNormKernel.swift",
                "Kernels/FlintRMSNormReference.swift",
                "Kernels/FlintRoPEKernel.swift",
                "Kernels/FlintRoPEReference.swift",
                "Kernels/FlintAttentionKernel.swift",
                "Kernels/FlintAttentionReference.swift",
                "Kernels/FlintRouterKernel.swift",
                "Kernels/FlintRouterReference.swift",
                "Kernels/FlintDequantizeKernel.swift",
                "Kernels/FlintDequantizeReference.swift",
                "Kernels/FlintReductionKernel.swift",
                "Kernels/FlintReductionReference.swift",
                "Kernels/FlintMLPKernel.swift",
                "Kernels/FlintMLPReference.swift",
            ],
            resources: [.copy("Metal")]
        ),
        .testTarget(
            name: "FlintFieldfareCoreTests",
            dependencies: ["FlintFieldfareCore"],
            path: "Tests/FlintFieldfareCoreTests",
            sources: [
                "FlintInferenceLeaseTests.swift",
                "FlintModelDirectoryTests.swift",
                "FlintSafetensorsFileTests.swift",
                "FlintResidentLoaderTests.swift",
                "FlintRMSNormKernelTests.swift",
                "FlintAttentionKernelTests.swift",
                "FlintRouterKernelTests.swift",
                "FlintDequantizeKernelTests.swift",
                "FlintReductionKernelTests.swift",
                "FlintMLPKernelTests.swift",
            ],
            resources: [.copy("Fixtures")]
        ),
    ]
)
