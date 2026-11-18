import Darwin
import Foundation

public enum FlintInferenceLeaseError: Error, LocalizedError, Equatable {
    case unavailable
    case inUse

    public var errorDescription: String? {
        switch self {
        case .unavailable: return "Cannot establish the local Flint inference ownership lock."
        case .inUse: return "Another FlintFieldfare app, server or benchmark owns inference. Stop or unload it before loading this model."
        }
    }
}

/// All Flint runner entry points acquire this lease before Metal or model allocation.
/// ARC retains the lease through runner lifetime and failed initialization.
/// The persistent lock file is never unlinked (unlink would allow two independently locked inodes).
/// Uses a dedicated lock file path to avoid conflict with QwenFieldfare or other tools.
public final class FlintInferenceLease: @unchecked Sendable {
    private let descriptor: Int32

    public init(path: String = "/tmp/flintfieldfare-inference-\(getuid()).lock") throws {
        let fd = open(path, O_CREAT | O_RDWR | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw FlintInferenceLeaseError.unavailable }
        do {
            var info = stat()
            guard fstat(fd, &info) == 0,
                  info.st_mode & S_IFMT == S_IFREG,
                  info.st_uid == getuid(),
                  info.st_mode & 0o077 == 0 else {
                throw FlintInferenceLeaseError.unavailable
            }
            guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
                throw FlintInferenceLeaseError.inUse
            }
            self.descriptor = fd
        } catch {
            close(fd)
            throw error
        }
    }

    deinit {
        close(descriptor)
    }
}
