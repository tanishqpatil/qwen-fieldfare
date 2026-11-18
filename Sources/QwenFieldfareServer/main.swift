import Foundation
import Darwin
import Metal
import QwenFieldfareServerSupport
import QwenFieldfareCore

@main struct Main {
    // Dispatch invokes these on its global queue; never inherit main-actor
    // isolation from the async executable entry point.
    nonisolated static func installSignals(_ shutdown: Cancellation) -> [DispatchSourceSignal] {
        signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN)
        return [SIGINT,SIGTERM].map { value in
            let source=DispatchSource.makeSignalSource(signal:value,queue:.global())
            source.setEventHandler {shutdown.cancel()};source.resume();return source
        }
    }
    static func main() async {
        do {
            let action=try ServerConfiguration.parse(Array(CommandLine.arguments.dropFirst()),environment:ProcessInfo.processInfo.environment)
            let config: ServerConfiguration
            switch action {
            case .help: print(ServerConfiguration.help);return
            case .version: print(ServerConfiguration.version);return
            case .serve(let value): config=value
            }
            let descriptor=open("/tmp/qwenfieldfare-server-\(getuid()).lock",O_CREAT|O_RDWR|O_NOFOLLOW|O_CLOEXEC,0o600)
            guard descriptor >= 0,flock(descriptor,LOCK_EX|LOCK_NB) == 0 else {throw APIError(503,"Another server owns inference")}
            defer {close(descriptor)}
            func memory() -> String {
                var info=task_vm_info_data_t();var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
                let code=withUnsafeMutablePointer(to:&info) {$0.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)}}
                return "footprint_bytes=\(code == KERN_SUCCESS ? info.phys_footprint:0), metal_bytes=\(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0)"
            }
            let shutdown=Cancellation()
            let signals=installSignals(shutdown)
            let started=Date()
            let backend=try await CoreBackend(path:config.modelPath,context:config.context,budget:config.budget,prefixCacheMode:config.prefixCacheMode)
            let server=try LocalServer(backend:backend,key:config.apiKey,model:config.modelID,queueCapacity:config.queueCapacity)
            let bound=try await server.start(host:config.host,port:config.port)
            print("QwenFieldfareServer ready 127.0.0.1:\(bound); load_seconds=\(Date().timeIntervalSince(started)); context=\(config.context), budget_gib=\(config.budget.rawValue), queue=\(config.queueCapacity), prefix_cache=\(config.prefixCacheMode.rawValue)")
            print("Loaded \(memory())");fflush(stdout)
            while !shutdown.isCancelled {try await Task.sleep(for:.milliseconds(100))}
            try await server.shutdown()
            print("Shutdown drained \(memory())");fflush(stdout)
            for signal in signals {signal.cancel()}
        } catch {
            let message: String
            if let error=error as? APIError {message=error.message}
            else if let error=error as? QwenInferenceLeaseError {message=error.localizedDescription}
            else {message="Startup/shutdown failed. Check model compatibility or whether the port is already in use."}
            fputs("QwenFieldfareServer: \(message)\n",stderr);exit(1)
        }
    }
}
