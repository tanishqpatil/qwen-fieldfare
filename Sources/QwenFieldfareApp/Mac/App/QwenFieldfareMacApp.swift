import AppKit
import QwenFieldfarePresentation
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let icon = MacAppIcon.load() {
            NSApp.applicationIconImage = icon
            NSApp.dockTile.display()
        }
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct QwenFieldfareMacApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var chatService = ChatService()

    var body: some Scene {
        Window("QwenFieldfare", id: "main") {
            RootView(chatService: chatService)
                .frame(minWidth: 1040, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1040, height: 720)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About QwenFieldfare") {
                    NSApp.orderFrontStandardAboutPanel(
                        options: AboutPanelPresentation.options(
                            infoDictionary: Bundle.main.infoDictionary,
                            icon: MacAppIcon.load()))
                }
            }
            CommandMenu("Generation") {
                Button("Cancel Generation") { chatService.cancelGeneration() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!chatService.isGenerating)
            }
            CommandMenu("Model") {
                Button("Choose Model Directory…", action: chooseModelDirectory)
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    .disabled(chatService.isGenerating || chatService.modelState == .loading || chatService.modelState == .unloading)
                Button("Unload Model", action: chatService.unloadModel)
                    .disabled(!chatService.canUnloadModel)
                Divider()
                Button("Reveal Model in Finder", action: revealModel)
                    .disabled(modelRevealTarget == .unavailable)
            }
        }
    }

    private var modelRevealTarget: ModelRevealTarget {
        ModelRevealPolicy.target(
            forModelPath: chatService.modelPath,
            fileExists: FileManager.default.fileExists(atPath:))
    }

    private func revealModel() {
        switch modelRevealTarget {
        case .selectItem(let url):
            NSWorkspace.shared.activateFileViewerSelecting([url])
        case .openContainer(let url):
            NSWorkspace.shared.open(url)
        case .unavailable:
            break
        }
    }

    private func chooseModelDirectory() {
        let panel = NSOpenPanel()
        panel.title = "Choose Qwen 3.6 35B-A3B Model Directory"
        panel.prompt = "Choose Model"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: chatService.modelPath, isDirectory: true)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try chatService.setModelPath(url.path)
        } catch {
            chatService.errorMessage = error.localizedDescription
        }
    }

}
