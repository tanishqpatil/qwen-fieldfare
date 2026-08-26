// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "QwenFieldfare",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "QwenFieldfareCore", targets: ["QwenFieldfareCore"]),
        .executable(name: "QwenFieldfareMac", targets: ["QwenFieldfareMac"]),
        .executable(name: "QwenFieldfareBench", targets: ["QwenFieldfareBench"]),
        .executable(name: "QwenFieldfarePack", targets: ["QwenFieldfarePack"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "QwenFieldfareCore",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            path: "Sources/QwenFieldfareCore",
            sources: [
                "Infrastructure/Metal/MetalContext.swift",
                "Infrastructure/ModelIO/QwenModelDirectory.swift",
                "Infrastructure/ModelIO/QwenResidentCoreLoader.swift",
                "Infrastructure/ModelIO/Sha256Verifier.swift",
                "Kernels/Attention/Attention.swift",
                "Kernels/LinearAttention/QwenDeltaNetKernel.swift",
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
            dependencies: ["QwenFieldfareCore", "QwenFieldfarePresentation"],
            path: "Sources/QwenFieldfareApp/Mac",
            sources: [
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
                "QwenGenerationSupportTests.swift",
                "QwenModelDirectoryTests.swift",
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
    ]
)
