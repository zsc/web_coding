let DDPG_CONFIG = {};
    // --- Math & NN Helpers ---
    function relu(x) { /* ... */ return Math.max(0, x); }
    function reluDerivative(x) { /* ... */ return x > 0 ? 1 : 0; }
    function tanh(x) { /* ... */ return Math.tanh(x); }
    function tanhDerivative(x) { /* ... */ return 1 - Math.tanh(x) * Math.tanh(x); }
    function dot(A, B) { /* ... */ if (!Array.isArray(A[0])) { return A.reduce((sum, val, i) => sum + val * B[i], 0); } return A.map(row => row.reduce((sum, val, i) => sum + val * B[i], 0)); }
    function addVectors(A, B) { /* ... */ return A.map((val, i) => val + B[i]); }
    function scaleVector(vec, scalar) { /* ... */ return vec.map(val => val * scalar); }
    function transpose(matrix) { /* ... */ if (!matrix || matrix.length === 0 || !matrix[0] || matrix[0].length === 0) return []; return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex])); }
    function initWeights(rows, cols, isReluLayer = true) { /* ... */ const weights = []; const limit = isReluLayer ? Math.sqrt(2 / cols) : Math.sqrt(1 / cols); for (let i = 0; i < rows; i++) { weights[i] = []; for (let j = 0; j < cols; j++) { weights[i][j] = (Math.random() - 0.5) * 2 * limit; } } return weights; }
    function initBias(size) { /* ... */ return new Array(size).fill(0).map(() => (Math.random() - 0.5) * 0.01); }

    // --- Neural Network Classes (2 Hidden Layers) ---
    class ActorNetwork {
        // Constructor, predict, getWeights, setWeights are the same as before
        constructor(stateDim, actionDim, hiddenDim1, hiddenDim2, actionBound) { this.actionBound = actionBound; this.w1 = initWeights(hiddenDim1, stateDim, true); this.b1 = initBias(hiddenDim1); this.w2 = initWeights(hiddenDim2, hiddenDim1, true); this.b2 = initBias(hiddenDim2); this.w3 = initWeights(actionDim, hiddenDim2, false); this.b3 = initBias(actionDim); }
        predict(state) { const h1_pre = addVectors(dot(this.w1, state), this.b1); const h1 = h1_pre.map(relu); const h2_pre = addVectors(dot(this.w2, h1), this.b2); const h2 = h2_pre.map(relu); const action_pre = addVectors(dot(this.w3, h2), this.b3); const action_unscaled = action_pre.map(tanh); return scaleVector(action_unscaled, this.actionBound); }
        getWeights() { return [JSON.parse(JSON.stringify(this.w1)), JSON.parse(JSON.stringify(this.b1)), JSON.parse(JSON.stringify(this.w2)), JSON.parse(JSON.stringify(this.b2)), JSON.parse(JSON.stringify(this.w3)), JSON.parse(JSON.stringify(this.b3))]; }
        setWeights(weights) { [this.w1, this.b1, this.w2, this.b2, this.w3, this.b3] = weights.map(w => JSON.parse(JSON.stringify(w))); }

        // update method now takes batched gradients
        update(states_batch, batched_dQda_scaled, lr) {
            // Initialize summed gradients for weights and biases
            let sum_dw1 = this.w1.map(row => row.map(() => 0)); let sum_db1 = this.b1.map(() => 0);
            let sum_dw2 = this.w2.map(row => row.map(() => 0)); let sum_db2 = this.b2.map(() => 0);
            let sum_dw3 = this.w3.map(row => row.map(() => 0)); let sum_db3 = this.b3.map(() => 0);

            const batchSize = states_batch.length;

            for (let b = 0; b < batchSize; b++) {
                const state = states_batch[b];
                const dQda_scaled = batched_dQda_scaled[b];

                // --- Forward pass for this sample to get intermediate activations ---
                const h1_pre = addVectors(dot(this.w1, state), this.b1);
                const h1 = h1_pre.map(relu);
                const h2_pre = addVectors(dot(this.w2, h1), this.b2);
                const h2 = h2_pre.map(relu);
                const action_unscaled_pre = addVectors(dot(this.w3, h2), this.b3);

                // --- Backpropagation for this sample ---
                let grad_L_wrt_output = dQda_scaled;
                grad_L_wrt_output = scaleVector(grad_L_wrt_output, this.actionBound);
                let grad_L_wrt_pre_activation = grad_L_wrt_output.map((g, i) => g * tanhDerivative(action_unscaled_pre[i]));

                // Layer 3 (Output w3, b3)
                const dw3_sample = [];
                for (let i = 0; i < this.w3.length; i++) { dw3_sample[i] = h2.map(h2_val => grad_L_wrt_pre_activation[i] * h2_val); }
                const db3_sample = [...grad_L_wrt_pre_activation];
                // Accumulate gradients
                for(let i=0; i<this.w3.length; i++) for(let j=0; j<this.w3[i].length; j++) sum_dw3[i][j] += dw3_sample[i][j];
                for(let i=0; i<this.b3.length; i++) sum_db3[i] += db3_sample[i];


                let grad_L_wrt_activation = dot(transpose(this.w3), grad_L_wrt_pre_activation);
                grad_L_wrt_pre_activation = grad_L_wrt_activation.map((g, i) => g * reluDerivative(h2_pre[i]));

                // Layer 2 (Hidden w2, b2)
                const dw2_sample = [];
                for (let i = 0; i < this.w2.length; i++) { dw2_sample[i] = h1.map(h1_val => grad_L_wrt_pre_activation[i] * h1_val); }
                const db2_sample = [...grad_L_wrt_pre_activation];
                for(let i=0; i<this.w2.length; i++) for(let j=0; j<this.w2[i].length; j++) sum_dw2[i][j] += dw2_sample[i][j];
                for(let i=0; i<this.b2.length; i++) sum_db2[i] += db2_sample[i];

                grad_L_wrt_activation = dot(transpose(this.w2), grad_L_wrt_pre_activation);
                grad_L_wrt_pre_activation = grad_L_wrt_activation.map((g, i) => g * reluDerivative(h1_pre[i]));

                // Layer 1 (Hidden w1, b1)
                const dw1_sample = [];
                for (let i = 0; i < this.w1.length; i++) { dw1_sample[i] = state.map(s_val => grad_L_wrt_pre_activation[i] * s_val); }
                const db1_sample = [...grad_L_wrt_pre_activation];
                for(let i=0; i<this.w1.length; i++) for(let j=0; j<this.w1[i].length; j++) sum_dw1[i][j] += dw1_sample[i][j];
                for(let i=0; i<this.b1.length; i++) sum_db1[i] += db1_sample[i];
            }

            // --- Apply mean updates (Gradient Ascent because maximizing Q) ---
            const invBatchSize = 1.0 / batchSize;
            for(let i=0; i<this.w1.length; i++) for(let j=0; j<this.w1[i].length; j++) this.w1[i][j] += lr * sum_dw1[i][j] * invBatchSize;
            for(let i=0; i<this.b1.length; i++) this.b1[i] += lr * sum_db1[i] * invBatchSize;
            for(let i=0; i<this.w2.length; i++) for(let j=0; j<this.w2[i].length; j++) this.w2[i][j] += lr * sum_dw2[i][j] * invBatchSize;
            for(let i=0; i<this.b2.length; i++) this.b2[i] += lr * sum_db2[i] * invBatchSize;
            for(let i=0; i<this.w3.length; i++) for(let j=0; j<this.w3[i].length; j++) this.w3[i][j] += lr * sum_dw3[i][j] * invBatchSize;
            for(let i=0; i<this.b3.length; i++) this.b3[i] += lr * sum_db3[i] * invBatchSize;
        }
    }

    class CriticNetwork {
        // Constructor, predict, getWeights, setWeights, get_dQ_da are the same as before
        constructor(stateDim, actionDim, hiddenDim1, hiddenDim2) { const inputDim = stateDim + actionDim; this.w1 = initWeights(hiddenDim1, inputDim, true); this.b1 = initBias(hiddenDim1); this.w2 = initWeights(hiddenDim2, hiddenDim1, true); this.b2 = initBias(hiddenDim2); this.w3 = initWeights(1, hiddenDim2, false); this.b3 = initBias(1); }
        predict(state, action) { const input = state.concat(action); const h1_pre = addVectors(dot(this.w1, input), this.b1); const h1 = h1_pre.map(relu); const h2_pre = addVectors(dot(this.w2, h1), this.b2); const h2 = h2_pre.map(relu); const q_value_pre = addVectors(dot(this.w3, h2), this.b3); return q_value_pre[0]; }
        getWeights() { return [JSON.parse(JSON.stringify(this.w1)), JSON.parse(JSON.stringify(this.b1)), JSON.parse(JSON.stringify(this.w2)), JSON.parse(JSON.stringify(this.b2)), JSON.parse(JSON.stringify(this.w3)), JSON.parse(JSON.stringify(this.b3))]; }
        setWeights(weights) { [this.w1, this.b1, this.w2, this.b2, this.w3, this.b3] = weights.map(w => JSON.parse(JSON.stringify(w))); }
        get_dQ_da(state, action) { const input = state.concat(action); const h1_pre = addVectors(dot(this.w1, input), this.b1); const h1 = h1_pre.map(relu); const h2_pre = addVectors(dot(this.w2, h1), this.b2); const h2 = h2_pre.map(relu); let dQ_dpre = [1.0]; let dQ_dh_activation = this.w3[0].map(w3_val => w3_val * dQ_dpre[0]); let dQ_dh_pre_activation = dQ_dh_activation.map((g, i) => g * reluDerivative(h2_pre[i])); dQ_dh_activation = dot(transpose(this.w2), dQ_dh_pre_activation); dQ_dh_pre_activation = dQ_dh_activation.map((g, i) => g * reluDerivative(h1_pre[i])); const dQ_da = new Array(DDPG_CONFIG.ACTION_DIM).fill(0); for (let j = 0; j < DDPG_CONFIG.ACTION_DIM; j++) { for (let i = 0; i < DDPG_CONFIG.HIDDEN_DIM1; i++) { dQ_da[j] += dQ_dh_pre_activation[i] * this.w1[i][DDPG_CONFIG.STATE_DIM + j]; } } return dQ_da; }

        update(state, action, target_q_value, lr) { // Backprop logic is the same for a single sample
            const input = state.concat(action); const h1_pre = addVectors(dot(this.w1, input), this.b1); const h1 = h1_pre.map(relu); const h2_pre = addVectors(dot(this.w2, h1), this.b2); const h2 = h2_pre.map(relu); const q_predicted_pre = addVectors(dot(this.w3, h2), this.b3); const q_predicted = q_predicted_pre[0];
            let grad_L_wrt_output = q_predicted - target_q_value;
            const dw3_0 = h2.map(h2_val => grad_L_wrt_output * h2_val); const db3_0 = grad_L_wrt_output;
            let grad_L_wrt_activation = this.w3[0].map(w3_val => w3_val * grad_L_wrt_output);
            let grad_L_wrt_pre_activation = grad_L_wrt_activation.map((g, i) => g * reluDerivative(h2_pre[i]));
            const dw2 = []; for (let i = 0; i < DDPG_CONFIG.HIDDEN_DIM2; i++) { dw2[i] = h1.map(h1_val => grad_L_wrt_pre_activation[i] * h1_val); } const db2 = [...grad_L_wrt_pre_activation];
            grad_L_wrt_activation = dot(transpose(this.w2), grad_L_wrt_pre_activation);
            grad_L_wrt_pre_activation = grad_L_wrt_activation.map((g, i) => g * reluDerivative(h1_pre[i]));
            const dw1 = []; for (let i = 0; i < DDPG_CONFIG.HIDDEN_DIM1; i++) { dw1[i] = input.map(input_val => grad_L_wrt_pre_activation[i] * input_val); } const db1 = [...grad_L_wrt_pre_activation];
            for(let j=0; j<this.w3[0].length; j++) this.w3[0][j] -= lr * dw3_0[j]; this.b3[0] -= lr * db3_0;
            for(let i=0; i<this.w2.length; i++) for(let j=0; j<this.w2[i].length; j++) this.w2[i][j] -= lr * dw2[i][j]; for(let i=0; i<this.b2.length; i++) this.b2[i] -= lr * db2[i];
            for(let i=0; i<this.w1.length; i++) for(let j=0; j<this.w1[i].length; j++) this.w1[i][j] -= lr * dw1[i][j]; for(let i=0; i<this.b1.length; i++) this.b1[i] -= lr * db1[i];
            return 0.5 * Math.pow(q_predicted - target_q_value, 2);
        }
    }

    class ReplayBuffer { /* ... */ constructor(capacity) { this.capacity = capacity; this.buffer = []; this.position = 0; } add(s, a, r, ns, d) { const exp = { s, a, r, ns, d }; if (this.buffer.length < this.capacity) this.buffer.push(exp); else this.buffer[this.position] = exp; this.position = (this.position + 1) % this.capacity; } sample(batchSize) { const batch = []; const len = this.buffer.length; for (let i = 0; i < batchSize; i++) batch.push(this.buffer[Math.floor(Math.random() * len)]); return batch; } size() { return this.buffer.length; } }

    class DDPGAgent {
        constructor(agentConfig) { /* ... same as before, initializes actor/critic etc. ... */ this.config = agentConfig; this.actor = new ActorNetwork(this.config.STATE_DIM, this.config.ACTION_DIM, this.config.HIDDEN_DIM1, this.config.HIDDEN_DIM2, this.config.ACTION_BOUND); this.critic = new CriticNetwork(this.config.STATE_DIM, this.config.ACTION_DIM, this.config.HIDDEN_DIM1, this.config.HIDDEN_DIM2); this.targetActor = new ActorNetwork(this.config.STATE_DIM, this.config.ACTION_DIM, this.config.HIDDEN_DIM1, this.config.HIDDEN_DIM2, this.config.ACTION_BOUND); this.targetCritic = new CriticNetwork(this.config.STATE_DIM, this.config.ACTION_DIM, this.config.HIDDEN_DIM1, this.config.HIDDEN_DIM2); this.targetActor.setWeights(this.actor.getWeights()); this.targetCritic.setWeights(this.critic.getWeights()); this.replayBuffer = new ReplayBuffer(this.config.BUFFER_SIZE); this.criticLossHistory = []; this.actorLossHistory = []; }
        getAction(state, addNoise = true) { /* ... */ let action = this.actor.predict(state); if (addNoise) { const noise = action.map(() => (Math.random() - 0.5) * 2 * this.config.NOISE_STDDEV * this.config.ACTION_BOUND); action = addVectors(action, noise); } return action.map(a => Math.max(-this.config.ACTION_BOUND, Math.min(this.config.ACTION_BOUND, a))); }
        updateTargetNetworks() { /* ... */ const softUpdate = (targetW, localW, tau) => targetW.map((targetL, layerIdx) => { const localL = localW[layerIdx]; return Array.isArray(targetL[0]) ? targetL.map((r, i) => r.map((v, j) => tau * localL[i][j] + (1 - tau) * v)) : targetL.map((v, i) => tau * localL[i] + (1 - tau) * v); }); this.targetActor.setWeights(softUpdate(this.targetActor.getWeights(), this.actor.getWeights(), this.config.TAU)); this.targetCritic.setWeights(softUpdate(this.targetCritic.getWeights(), this.critic.getWeights(), this.config.TAU));}

        learn() {
            if (this.replayBuffer.size() < this.config.BATCH_SIZE) return { actorLoss: null, criticLoss: null};
            const batch = this.replayBuffer.sample(this.config.BATCH_SIZE);
            const states = batch.map(e => e.s); const actions = batch.map(e => e.a);
            const rewards = batch.map(e => e.r); const nextStates = batch.map(e => e.ns);
            const dones = batch.map(e => e.d);

            // --- Critic Update ---
            const targetNextActions = nextStates.map(s_prime => this.targetActor.predict(s_prime));
            const targetQValues = [];
            for (let i = 0; i < this.config.BATCH_SIZE; i++) {
                const q_prime = this.targetCritic.predict(nextStates[i], targetNextActions[i]);
                targetQValues.push(rewards[i] + this.config.GAMMA * (1 - dones[i]) * q_prime);
            }
            
            let batchCriticLoss = 0;
            for (let i = 0; i < this.config.BATCH_SIZE; i++) {
                const sampleCriticLoss = this.critic.update(states[i], actions[i], targetQValues[i], this.config.LR_CRITIC);
                batchCriticLoss += sampleCriticLoss;
            }
            const avgCriticLoss = batchCriticLoss / this.config.BATCH_SIZE;
            this.criticLossHistory.push(avgCriticLoss);
            
            // --- Actor Update (Batched) ---
            const actor_actions_for_grad = states.map(s_sample => this.actor.predict(s_sample));
            const batched_dQda = [];
            let batchActorLossProxy = 0;

            for (let i = 0; i < this.config.BATCH_SIZE; i++) {
                const s_sample = states[i];
                const actor_action_sample = actor_actions_for_grad[i];
                const dQda_sample = this.critic.get_dQ_da(s_sample, actor_action_sample);
                batched_dQda.push(dQda_sample);
                
                // Calculate actor "loss" proxy for this sample: -Q(s, actor(s))
                const q_val_for_actor_loss = this.critic.predict(s_sample, actor_action_sample);
                batchActorLossProxy += (-q_val_for_actor_loss);
            }
            const avgActorLossProxy = batchActorLossProxy / this.config.BATCH_SIZE;

            // Check if any dQda in the batch is problematic
            const isAnyGradProblematic = batched_dQda.some(grads => grads.some(g => isNaN(g) || !isFinite(g)));

            if (!isAnyGradProblematic) {
                this.actor.update(states, batched_dQda, this.config.LR_ACTOR); // Pass batched states and gradients
                this.actorLossHistory.push(avgActorLossProxy);
            } else {
                console.warn("NaN/Inf in dQ/da for at least one sample in actor update batch. Skipping actor update this step.");
                this.actorLossHistory.push(this.actorLossHistory.length > 0 ? this.actorLossHistory[this.actorLossHistory.length-1] : 0);
            }
            
            this.updateTargetNetworks();
            return { actorLoss: avgActorLossProxy, criticLoss: avgCriticLoss };
        }
    }

    class InvertedPendulumEnv { /* ... same as before ... */ constructor(envParams) { this.params = envParams; this.gravity = this.params.gravity; this.masscart = this.params.masscart; this.masspole = this.params.masspole; this.total_mass = this.masspole + this.masscart; this.length = this.params.length; this.polemass_length = this.masspole * this.length; this.force_mag = this.params.action_bound; this.tau = this.params.tau; this.theta_threshold_radians = this.params.theta_threshold_radians; this.x_threshold = this.params.x_threshold; this.state = null; } reset() { this.state = [0.0, 0.0, (Math.random() - 0.5) * 0.02, (Math.random() - 0.5) * 0.02]; return [...this.state]; } step(actionInput) { let [x_curr, x_dot_curr, theta_curr, theta_dot_curr] = this.state; let force = actionInput[0]; if (isNaN(force) || !isFinite(force)) { console.error(`Env: NaN/Infinity force: ${force}. Using 0.`); force = 0.0; } force = Math.max(-this.force_mag, Math.min(this.force_mag, force)); if (this.state.some(v => isNaN(v) || !isFinite(v))) { console.error(`Env: NaN/Inf in state before step: ${this.state}. Resetting.`); this.state = [0,0,0,0]; return { nextState: [...this.state], reward: -100.0, done: true }; } const costheta = Math.cos(theta_curr); const sintheta = Math.sin(theta_curr); if (isNaN(costheta) || isNaN(sintheta)) { this.state = [0,0,0,0]; return { nextState: [...this.state], reward: -100.0, done: true };} const pml_tds_sin = this.polemass_length * theta_dot_curr * theta_dot_curr * sintheta; const temp = (force + pml_tds_sin) / this.total_mass; if (isNaN(temp) || !isFinite(temp)) { this.state = [0,0,0,0]; return { nextState: [...this.state], reward: -100.0, done: true };} const denominator = this.length * (4.0/3.0 - this.masspole * costheta * costheta / this.total_mass); if (isNaN(denominator) || Math.abs(denominator) < 1e-9) { this.state = [0,0,0,0]; return { nextState: [...this.state], reward: -100.0, done: true };} const thetaacc = (this.gravity * sintheta - costheta * temp) / denominator; const xacc  = temp - this.polemass_length * thetaacc * costheta / this.total_mass; if (isNaN(thetaacc) || !isFinite(thetaacc) || isNaN(xacc) || !isFinite(xacc)) { this.state = [0,0,0,0]; return { nextState: [...this.state], reward: -100.0, done: true };} this.state = [ x_curr + this.tau * x_dot_curr, x_dot_curr + this.tau * xacc, theta_curr + this.tau * theta_dot_curr, theta_dot_curr + this.tau * thetaacc ]; if (this.state.some(v => isNaN(v) || !isFinite(v))) { console.error(`Env: NaN/Inf in state AFTER update: ${this.state}. Resetting.`); this.state = [0,0,0,0]; return { nextState: [...this.state], reward: -100.0, done: true }; } const done = this.state[0] < -this.x_threshold || this.state[0] > this.x_threshold || this.state[2] < -this.theta_threshold_radians || this.state[2] > this.theta_threshold_radians; let reward_value; if (!done) { const angle_norm = this.state[2] / this.theta_threshold_radians; const pos_norm = this.state[0] / this.x_threshold; reward_value = (isNaN(angle_norm) || isNaN(pos_norm)) ? -100.0 : (1.0 - 0.5 * Math.abs(angle_norm) - 0.1 * Math.abs(pos_norm)); } else { reward_value = -10.0; } if (isNaN(reward_value)) { console.error("Reward NaN!"); reward_value = -100.0; } return { nextState: [...this.state], reward: reward_value, done }; } render() { ctx.clearRect(0, 0, canvas.width, canvas.height); if (!this.state || this.state.some(isNaN)) { ctx.fillText("Error: Invalid state for rendering", 10, 20); return; } const [x, _, theta] = this.state; const cartWidth = 50, cartHeight = 30, cartY = canvas.height - 100; const cartX = x * (canvas.width / (2 * this.x_threshold)) + canvas.width / 2 - cartWidth / 2; ctx.fillStyle = 'blue'; ctx.fillRect(cartX, cartY, cartWidth, cartHeight); const poleRenderLength = 100; const poleX1 = cartX + cartWidth / 2, poleY1 = cartY; const poleX2 = poleX1 + poleRenderLength * Math.sin(theta); const poleY2 = poleY1 - poleRenderLength * Math.cos(theta); ctx.strokeStyle = 'red'; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(poleX1, poleY1); ctx.lineTo(poleX2, poleY2); ctx.stroke(); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, cartY + cartHeight); ctx.lineTo(canvas.width, cartY + cartHeight); ctx.stroke(); } }

module.exports = { ActorNetwork, CriticNetwork, ReplayBuffer, DDPGAgent, InvertedPendulumEnv, configure(c) { DDPG_CONFIG = c; } };
