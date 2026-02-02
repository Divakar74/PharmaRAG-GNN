// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DDIAuditTrail {
    struct PredictionRecord {
        string drugPair;
        uint256 probability;
        string modelHash;
        string userId;
        uint256 timestamp;
        string riskCategory;
    }
    
    PredictionRecord[] public records;
    mapping(string => uint256[]) public userHistory;
    
    event PredictionLogged(uint256 indexed recordId, string drugPair, uint256 probability, string modelHash, string userId, uint256 timestamp);
    
    function logPrediction(string memory drugPair, uint256 probability, string memory modelHash, string memory userId, string memory riskCategory) public returns (uint256) {
        uint256 recordId = records.length;
        records.push(PredictionRecord({drugPair: drugPair, probability: probability, modelHash: modelHash, userId: userId, timestamp: block.timestamp, riskCategory: riskCategory}));
        userHistory[userId].push(recordId);
        emit PredictionLogged(recordId, drugPair, probability, modelHash, userId, block.timestamp);
        return recordId;
    }
    
    function getRecordCount() public view returns (uint256) {
        return records.length;
    }
}
