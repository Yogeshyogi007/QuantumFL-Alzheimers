'use strict';
const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const DATA_FILE = path.join(__dirname, 'logs.json');

function loadMockData() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const text = fs.readFileSync(DATA_FILE, 'utf8');
      const parsed = JSON.parse(text);
      return Array.isArray(parsed) ? parsed : [];
    }
  } catch (_) {}
  return [];
}

function saveMockData(data) {
  try {
    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2), 'utf8');
  } catch (_) {}
}

const app = express();
app.use(bodyParser.json());

// Mock data store for testing
let mockData = loadMockData();

// Mock functions for testing without Fabric
async function getContract() {
  return {
    gateway: { disconnect: () => {} },
    contract: {
      submitTransaction: async (method, ...args) => {
        const key = `${args[4]}:${args[2]}:${args[0]}`;
        const record = {
          updateHash: args[0],
          accuracy: parseFloat(args[1]),
          hospitalId: args[2],
          storageUri: args[3],
          roundId: args[4],
          timestamp: Date.now()
        };
        mockData.push({ key, ...record });
        saveMockData(mockData);
        return Buffer.from(key);
      },
      evaluateTransaction: async (method) => {
        return Buffer.from(JSON.stringify(mockData));
      }
    }
  };
}

app.post('/record', async (req, res) => {
  try {
    const { updateHash, accuracy, hospitalId, storageUri, roundId } = req.body || {};
    if (!updateHash || !hospitalId || !roundId) return res.status(400).json({ error: 'Missing fields' });
    const { gateway, contract } = await getContract();
    const result = await contract.submitTransaction('recordModelUpdate', updateHash, String(accuracy||0), hospitalId, storageUri||'', String(roundId));
    await gateway.disconnect();
    res.json({ ok: true, key: result.toString() });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/history', async (req, res) => {
  try {
    const { gateway, contract } = await getContract();
    const result = await contract.evaluateTransaction('getAllUpdates');
    await gateway.disconnect();
    res.json(JSON.parse(result.toString()));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Fabric REST listening on :${PORT}`));


