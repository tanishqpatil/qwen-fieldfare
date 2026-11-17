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
            var options:[String:String]=[:];var args=CommandLine.arguments.dropFirst().makeIterator()
            while let key=args.next() {guard ["--model","--port","--context","--memory-gib","--model-id"].contains(key),let value=args.next() else {throw APIError(400,"Expected --model PATH [--port 8080 --context 8192 --memory-gib 5 --model-id qwen-fieldfare]")};options[key]=value}
            guard let path=options["--model"],let secret=ProcessInfo.processInfo.environment["QWENFIELDFARE_API_KEY"],let port=Int(options["--port"] ?? "8080"),let context=Int(options["--context"] ?? "8192"),let budget=QwenActiveStreamingMemoryBudget(rawValue:Int(options["--memory-gib"] ?? "5") ?? -1) else {throw APIError(400,"Require model path, valid settings and QWENFIELDFARE_API_KEY")}
            guard secret.utf8.count >= 16, !secret.contains("\n"), (0...65535).contains(port) else {throw APIError(400,"Invalid authentication/port configuration")}
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
            let backend=try await CoreBackend(path:path,context:context,budget:budget)
            let server=try LocalServer(backend:backend,key:secret,model:options["--model-id"] ?? "qwen-fieldfare")
            let bound=try await server.start(port:port)
            print("QwenFieldfareServer ready 127.0.0.1:\(bound); load_seconds=\(Date().timeIntervalSince(started)); context=\(context), budget_gib=\(budget.rawValue)")
            print("Loaded \(memory())");fflush(stdout)
            while !shutdown.isCancelled {try await Task.sleep(for:.milliseconds(100))}
            try await server.shutdown()
            print("Shutdown drained \(memory())");fflush(stdout)
            for signal in signals {signal.cancel()}
        } catch {fputs("QwenFieldfareServer startup/shutdown failed. Check local configuration and model compatibility.\n",stderr);exit(1)}
    }
}
