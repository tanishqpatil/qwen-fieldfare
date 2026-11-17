import Darwin
import Foundation

public enum QwenInferenceLeaseError: Error, LocalizedError, Equatable {
    case unavailable
    case inUse
    public var errorDescription: String? {
        switch self {
        case .unavailable: return "Cannot establish the local inference ownership lock."
        case .inUse: return "Another QwenFieldfare app, server or benchmark owns inference. Stop or unload it before loading this model."
        }
    }
}
/// All Q4 runner entry points acquire before Metal allocation. ARC retains the
/// lease through runner lifetime and failed initialization. The persistent lock
/// file is never unlinked (unlink would allow two independently locked inodes).
/// This excludes cooperating QwenFieldfare processes, not MLX/Ollama/etc.
final class QwenInferenceLease {
    private let descriptor: Int32
    init(path: String = "/tmp/qwenfieldfare-inference-\(getuid()).lock") throws {
        let fd=open(path,O_CREAT|O_RDWR|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC,0o600)
        guard fd >= 0 else {throw QwenInferenceLeaseError.unavailable}
        do {
            var info=stat()
            guard fstat(fd,&info) == 0,info.st_mode & S_IFMT == S_IFREG,info.st_uid == getuid(),info.st_mode & 0o077 == 0 else {throw QwenInferenceLeaseError.unavailable}
            guard flock(fd,LOCK_EX|LOCK_NB) == 0 else {throw QwenInferenceLeaseError.inUse}
            descriptor=fd
        } catch {close(fd);throw error}
    }
    deinit {close(descriptor)}
}
