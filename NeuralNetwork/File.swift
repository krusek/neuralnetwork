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

public enum DataName {
    case t10k, train

    var labelFilename: String {
        switch self {
        case .t10k:
            return "t10k-labels-idx1-ubyte"
        case .train:
            return "train-labels-idx1-ubyte"
        }
    }

    var imageFilename: String {
        switch self {
        case .t10k:
            return "t10k-images-idx3-ubyte"
        case .train:
            return "train-images-idx3-ubyte"
        }
    }
}

public class Stream {
    let data: Data
    var index: Data.Index
    init?(_ name: String) {
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

    public static func labels(_ name: DataName) -> Stream? {
        return Stream(name.labelFilename)
    }

    public static func images(_ name: DataName) -> Stream? {
        return Stream(name.imageFilename)
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
        self.index = end
        return pointer.first ?? 0
    }

    public static func readTrainingData(_ type: DataName) throws -> [(PhotoData, Output<Double>)] {
        guard let images = Stream.images(type), let labels = Stream.images(type) else {
            throw "Files not found."
        }

        _ = images.readInt()
        _ = labels.readInt()
        let size1 = images.readInt()
        let size2 = labels.readInt()
        guard size1 == size2 else {
            throw "Files of different sizes! images: \(size1), labels: \(size2)"
        }

        let rows = images.readInt()
        let cols = images.readInt()
        return (0..<size1).map { (_) -> (PhotoData, Output<Double>) in
            let data = (0..<(rows * cols)).map { _ in images.readByte() }
            let expected = labels.readByte()
            return  (data, intToArray(expected))
        }
    }
}

extension String: Error {}

func intToArray(_ int: UInt8) -> [Double] {
    var array = Array<Double>(repeating: 0.0, count: 10)
    guard int < 10 && int >= 0 else { return array }
    array[Int(int)] = 1.0
    return array
}
