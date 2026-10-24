// Round-trip PyTorch weights into the exact upstream TensorFlow.js layer layout.
const fs=require('node:fs');
const tf=require('../vendor/tf-4.22.0.min.js');
const [modelPath='results/remote/cuda_graph/model.json',fixturePath='results/parity_fixture.json',
  exportDir='exports/tfjs',resultPath='results/tfjs_parity.json']=process.argv.slice(2);
(async()=>{
  await tf.setBackend('cpu');await tf.ready();
  const raw=JSON.parse(fs.readFileSync(modelPath));
  const fixture=JSON.parse(fs.readFileSync(fixturePath));
  const a=raw.architecture;const model=tf.sequential();
  model.add(tf.layers.inputLayer({inputShape:[28,28,1]}));
  for(const filters of a.filters){
    model.add(tf.layers.conv2d({filters,kernelSize:a.kernel,activation:'relu',padding:'same'}));
    model.add(tf.layers.maxPooling2d({poolSize:a.pool,strides:a.pool}));
  }
  model.add(tf.layers.flatten());model.add(tf.layers.dense({units:a.dense,activation:'relu'}));
  model.add(tf.layers.dense({units:10,activation:'softmax'}));
  const tensor=k=>tf.tensor(raw.tensors[k].data,raw.tensors[k].shape);
  const weights=[];
  a.filters.forEach((_,i)=>weights.push(tensor(`convs.${i}.weight`).transpose([2,3,1,0]),tensor(`convs.${i}.bias`)));
  for(const n of ['fc','out']) weights.push(tensor(n+'.weight').transpose(),tensor(n+'.bias'));
  model.setWeights(weights);
  const x=tf.tensor(fixture.pixels).reshape([fixture.indices.length,28,28,1]).div(255);
  const probabilities=await model.predict(x).array();
  let max_abs=0,agreement=0;
  const argmax=a=>a.indexOf(Math.max(...a));
  probabilities.forEach((row,i)=>{row.forEach((v,j)=>{max_abs=Math.max(max_abs,Math.abs(v-fixture.probabilities[i][j]))});agreement+=argmax(row)===argmax(fixture.probabilities[i])});
  const result={tfjs:tf.version.tfjs,backend:tf.getBackend(),samples:fixture.indices.length,
    max_probability_absolute_error:max_abs,prediction_agreement:agreement/probabilities.length,
    passed:max_abs<2e-5&&agreement===probabilities.length,indices:fixture.indices};
  fs.mkdirSync(exportDir,{recursive:true});
  await model.save(tf.io.withSaveHandler(async artifacts=>{
    fs.writeFileSync(exportDir+'/weights.bin',Buffer.from(artifacts.weightData));
    fs.writeFileSync(exportDir+'/model.json',JSON.stringify({format:'layers-model',generatedBy:`TensorFlow.js ${tf.version.tfjs}`,
      convertedBy:'mnist_gpu_lab: OIHW→HWIO; Dense transpose; NHWC flatten',modelTopology:artifacts.modelTopology,
      weightsManifest:[{paths:['weights.bin'],weights:artifacts.weightSpecs}]},null,2));
    return {modelArtifactsInfo:{dateSaved:new Date(),modelTopologyType:'JSON'}};
  }));
  fs.writeFileSync(resultPath,JSON.stringify(result,null,2));console.log(result);
  if(!result.passed)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
