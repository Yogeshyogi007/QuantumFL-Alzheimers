'use strict';
const { Contract } = require('fabric-contract-api');

class QFLUpdatesContract extends Contract {
	async Init(ctx) { return 'OK'; }

	async recordModelUpdate(ctx, updateHash, accuracy, hospitalId, storageUri, roundId) {
		const key = `${roundId}:${hospitalId}:${updateHash}`;
		const exists = await ctx.stub.getState(key);
		if (exists && exists.length) throw new Error('Update already exists');
		const update = {
			updateHash,
			accuracy: parseInt(accuracy, 10),
			hospitalId,
			storageUri,
			roundId,
			timestamp: Date.now()
		};
		await ctx.stub.putState(key, Buffer.from(JSON.stringify(update)));
		return key;
	}

	async getModelUpdate(ctx, key) {
		const data = await ctx.stub.getState(key);
		if (!data || !data.length) throw new Error('Not found');
		return data.toString('utf8');
	}

	async getAllUpdates(ctx) {
		const iter = await ctx.stub.getStateByRange('', '');
		const results = [];
		for await (const res of iter) {
			results.push({ key: res.key, ...JSON.parse(res.value.toString('utf8')) });
		}
		return JSON.stringify(results);
	}
}

module.exports = QFLUpdatesContract;

