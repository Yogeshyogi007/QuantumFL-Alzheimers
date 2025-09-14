# Private Fabric setup with Minifab (for QFL model logging)

Prerequisites:
- Docker & Docker Compose
- Node.js >= 14 and npm
- Git

Bring up Fabric with Minifab:
- curl -sL https://tinyurl.com/minifab | bash
- ./minifab up -i 2.5 -o org0.example.com -s couchdb -l node
- ./minifab create -c qflchannel
- ./minifab join -c qflchannel
- ./minifab anchorupdate

Deploy JavaScript chaincode:
- ./minifab ccup -n qflupdates -l node -v 1.0 -p fabric/chaincode/qflupdates -C qflchannel

Start REST service:
- cd fabric/service
- npm install
- Set envs and run:
  - FABRIC_CONNECTION_JSON=connection.json
  - FABRIC_WALLET=./wallet
  - FABRIC_CHANNEL=qflchannel
  - FABRIC_CC=qflupdates
  - FABRIC_ID=appUser
  - node server.js

REST endpoints:
- POST /record { updateHash, accuracy, hospitalId, storageUri, roundId }
- GET /history returns array of updates

Wire Flask:
- set FABRIC_RECORD_URL=http://localhost:3000/record
- set FABRIC_HISTORY_URL=http://localhost:3000/history
- set FABRIC_STORAGE_URI=file://<absolute_path_to_models>

The Flask trainer will POST on job completion with the model hash and metadata.
