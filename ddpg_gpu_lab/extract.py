"""Extract the computational classes byte-for-byte; no DOM or training UI."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
source = (root / 'source/original.html').read_text()
core = source[source.index('    // --- Math & NN Helpers ---'):source.index('    function init()')]
(root / 'core_original.cjs').write_text('let DDPG_CONFIG = {};\n' + core + '\nmodule.exports = { ActorNetwork, CriticNetwork, ReplayBuffer, DDPGAgent, InvertedPendulumEnv, configure(c) { DDPG_CONFIG = c; } };\n')
assert hashlib.sha1(b'blob '+str(len(source.encode())).encode()+b'\0'+source.encode()).hexdigest() == '6409582ccaf159a822a422bec7f8fe2ed9fda144', 'Source snapshot differs from the recorded Git blob'
(root / 'source/provenance.json').write_text(json.dumps({'url':'https://github.com/zsc/zsc.github.io/blob/master/widgets/ddpg_invpendulum.html','git_blob_sha':'6409582ccaf159a822a422bec7f8fe2ed9fda144','snapshot_sha256':hashlib.sha256(source.encode()).hexdigest(),'retrieved_date':'2026-10-24','extraction':'Math & NN Helpers through end of InvertedPendulumEnv; render method retained but never invoked'},indent=2))
