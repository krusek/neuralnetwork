//
//  File.swift
//  NeuralNetwork
//
//  Created by Korben Rusek on 1/16/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import Foundation

class MData {
    private func openFile(_ name: String) {
        
    }
}

public class Stream {
    let data: Data
    var index: Data.Index
    public init?(_ name: String) {
        let path = "/Users/korbenrusek/Documents/code/neural/neuralnetwork"
        let filePath = "\(path)/data/\(name)"
        print(filePath)
        let exists = FileManager.default.fileExists(atPath: filePath)
        print("exists: \(exists)")
        if let data = FileManager.default.contents(atPath: filePath) {
            self.data = data
            self.index = data.startIndex
        } else {
            return nil
        }
    }

    public func readInt() -> Int32 {
        let pointer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 4)
        let end = index.advanced(by: 4)
        data.copyBytes(to: pointer, from: Range<Data.Index>(uncheckedBounds: (index, end)))
        var int: Int32 = 0
        for item in pointer {
            int = (int << 8) | Int32(item)
        }
        self.index = end
        return int
    }

    public func readByte() -> UInt8 {
        let pointer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 1)
        let end = index.advanced(by: 1)
        data.copyBytes(to: pointer, from: Range<Data.Index>(uncheckedBounds: (index, end)))
//        var int: Int32 = 0
//        for item in pointer {
//            int = (int << 8) | Int32(item)
//        }
        self.index = end
        return pointer.first ?? 0
    }
}
