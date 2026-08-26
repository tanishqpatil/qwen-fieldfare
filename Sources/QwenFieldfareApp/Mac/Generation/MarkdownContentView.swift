import AppKit
import SwiftUI
import QwenFieldfarePresentation

struct MarkdownContentView: View {
    let content: String
    let isStreaming: Bool

    var body: some View {
        let parsedBlocks = FastMarkdownParser.parse(content, isStreaming: isStreaming)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(parsedBlocks) { block in
                switch block.kind {
                case .thought(let thoughtText, let isStillThinking):
                    ThoughtProcessView(thought: thoughtText, isStreaming: isStillThinking)
                case .code(let language, let code):
                    CodeBlockView(language: language, code: code)
                case .table(let headers, let rows):
                    MarkdownTableView(headers: headers, rows: rows)
                case .heading(let level, let text):
                    MarkdownHeadingView(level: level, text: text)
                case .quote(let text):
                    MarkdownQuoteView(text: text)
                case .prose(let text):
                    MarkdownProseView(text: text)
                }
            }

            if isStreaming && !isCurrentlyInThought(parsedBlocks) {
                StreamingCursorView()
            }
        }
        .textSelection(.enabled)
    }

    private func isCurrentlyInThought(_ blocks: [ParsedContentBlock]) -> Bool {
        if case .thought(_, true) = blocks.last?.kind {
            return true
        }
        return false
    }
}

// MARK: - Markdown Table View

struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 0) {
                // Header Row
                HStack(spacing: 0) {
                    ForEach(headers.indices, id: \.self) { colIdx in
                        Text(LocalizedStringKey(headers[colIdx]))
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(Color(nsColor: .labelColor))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .frame(minWidth: columnWidth(colIdx), alignment: .leading)

                        if colIdx < headers.count - 1 {
                            Divider()
                        }
                    }
                }
                .background(Color(nsColor: .controlBackgroundColor).opacity(0.85))

                Divider()

                // Data Rows
                ForEach(rows.indices, id: \.self) { rowIdx in
                    let row = rows[rowIdx]
                    HStack(spacing: 0) {
                        ForEach(headers.indices, id: \.self) { colIdx in
                            let cellText = colIdx < row.count ? row[colIdx] : ""
                            Text(LocalizedStringKey(cellText))
                                .font(.system(size: 13))
                                .foregroundStyle(Color(nsColor: .labelColor))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .frame(minWidth: columnWidth(colIdx), alignment: .leading)

                            if colIdx < headers.count - 1 {
                                Divider()
                            }
                        }
                    }
                    .background(
                        rowIdx.isMultiple(of: 2)
                            ? Color(nsColor: .textBackgroundColor).opacity(0.3)
                            : Color.clear
                    )

                    if rowIdx < rows.count - 1 {
                        Divider()
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 1)
            )
        }
        .padding(.vertical, 4)
    }

    private func columnWidth(_ index: Int) -> CGFloat {
        100
    }
}

// MARK: - Markdown Headings

private struct MarkdownHeadingView: View {
    let level: Int
    let text: String

    var body: some View {
        Text(LocalizedStringKey(text))
            .font(headingFont)
            .foregroundStyle(Color(nsColor: .labelColor))
            .padding(.top, level == 1 ? 8 : (level == 2 ? 6 : 4))
            .padding(.bottom, 2)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var headingFont: Font {
        switch level {
        case 1: return .system(size: 19, weight: .bold)
        case 2: return .system(size: 16, weight: .bold)
        case 3: return .system(size: 14.5, weight: .semibold)
        default: return .system(size: 13.5, weight: .semibold)
        }
    }
}

// MARK: - Markdown Blockquote View

private struct MarkdownQuoteView: View {
    let text: String

    var body: some View {
        HStack(spacing: 10) {
            RoundedRectangle(cornerRadius: 2)
                .fill(Color.accentColor.opacity(0.7))
                .frame(width: 3)

            Text(LocalizedStringKey(text))
                .font(.system(size: 13.5))
                .italic()
                .foregroundStyle(.secondary)
                .lineSpacing(3)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
        .padding(.leading, 2)
    }
}

// MARK: - Markdown Prose View

private struct MarkdownProseView: View {
    let text: String

    var body: some View {
        Text(LocalizedStringKey(text))
            .font(.system(size: 14))
            .lineSpacing(4)
            .foregroundStyle(Color(nsColor: .labelColor))
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Expandable Thought Process View

struct ThoughtProcessView: View {
    let thought: String
    let isStreaming: Bool
    @State private var isExpanded: Bool

    init(thought: String, isStreaming: Bool) {
        self.thought = thought
        self.isStreaming = isStreaming
        // Default to expanded while streaming thinking, or collapsed when finished
        _isExpanded = State(initialValue: isStreaming)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Clickable Header Button
            Button {
                withAnimation(.spring(duration: 0.25)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.purple)

                    Text(headerTitle)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.primary)

                    if isStreaming {
                        ProgressView().controlSize(.mini)
                        Text("Thinking…")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            // Expanded Thought Body
            if isExpanded {
                Divider()
                    .background(Color.purple.opacity(0.2))

                ScrollView {
                    Text(thought)
                        .font(.system(size: 12.5, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineSpacing(3)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(maxHeight: 260)
                .background(Color(nsColor: .textBackgroundColor).opacity(0.3))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.purple.opacity(0.06))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.purple.opacity(0.18), lineWidth: 1)
                )
        )
        .padding(.vertical, 2)
    }

    private var headerTitle: String {
        let words = thought.split(separator: " ").count
        return isExpanded ? "Thought Process" : "Thought Process (\(words) words)"
    }
}

// MARK: - Streaming Cursor View

private struct StreamingCursorView: View {
    @State private var opacity: Double = 0.2

    var body: some View {
        Circle()
            .fill(Color.accentColor)
            .frame(width: 7, height: 7)
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true)) {
                    opacity = 1.0
                }
            }
            .padding(.vertical, 2)
    }
}
