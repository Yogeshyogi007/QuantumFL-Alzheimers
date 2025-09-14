// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ModelUpdates {
    struct UpdateLog {
        bytes32 updateHash; // SHA256 hash of model weights
        uint256 accuracyBp; // accuracy in basis points (e.g., 9876 => 98.76%)
        string hospitalId;  // Hospital identifier
        uint256 timestamp;  // Block timestamp
    }

    UpdateLog[] public logs;

    event UpdateRecorded(bytes32 indexed updateHash, uint256 accuracyBp, string hospitalId, uint256 timestamp);

    function recordUpdate(bytes32 updateHash, uint256 accuracyBp, string calldata hospitalId) external {
        UpdateLog memory logItem = UpdateLog({
            updateHash: updateHash,
            accuracyBp: accuracyBp,
            hospitalId: hospitalId,
            timestamp: block.timestamp
        });
        logs.push(logItem);
        emit UpdateRecorded(updateHash, accuracyBp, hospitalId, block.timestamp);
    }

    function getUpdateCount() external view returns (uint256) {
        return logs.length;
    }

    function getUpdate(uint256 idx) external view returns (bytes32, uint256, string memory, uint256) {
        UpdateLog memory u = logs[idx];
        return (u.updateHash, u.accuracyBp, u.hospitalId, u.timestamp);
    }
}


