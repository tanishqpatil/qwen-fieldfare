import Foundation
import Security

@MainActor public protocol ServerCredentialStore {
    func read() throws -> String?
    func write(_ key: String) throws
}
@MainActor public final class KeychainServerCredential: ServerCredentialStore {
    private let service = "QwenFieldfare.local-server"
    public init() {}
    private var query: [String:Any] {[kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:"default"]}
    public func read() throws -> String? {
        var query = query;query[kSecReturnData as String] = true;query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary,&result)
        if status == errSecItemNotFound {return nil}
        guard status == errSecSuccess,let data = result as? Data,let key = String(data:data,encoding:.utf8) else {throw ManagementError.credential}
        return key
    }
    public func write(_ key: String) throws {
        let attributes: [String:Any] = [kSecValueData as String:Data(key.utf8)]
        let status = SecItemUpdate(query as CFDictionary,attributes as CFDictionary)
        if status == errSecItemNotFound {
            var item = query;item.merge(attributes) {_,new in new};item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            guard SecItemAdd(item as CFDictionary,nil) == errSecSuccess else {throw ManagementError.credential}
        } else if status != errSecSuccess {throw ManagementError.credential}
    }
    static func generate() throws -> String {
        var bytes = [UInt8](repeating:0,count:32)
        guard SecRandomCopyBytes(kSecRandomDefault,bytes.count,&bytes) == errSecSuccess else {throw ManagementError.credential}
        return bytes.map {String(format:"%02x",$0)}.joined()
    }
}
