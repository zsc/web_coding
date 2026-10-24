const fs=require('node:fs');
const {chromium}=require('/Users/zsc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page=await browser.newPage({viewport:{width:1440,height:1040},deviceScaleFactor:1});
  const errors=[],external=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(/^https?:/.test(r.url()))external.push(r.url())});
  await page.goto('file://'+process.cwd()+'/report.html');
  await page.waitForFunction(()=>document.querySelectorAll('.sample').length===12);
  const initial=await page.evaluate(()=>({time:document.querySelector('#heroTime').textContent,speed:document.querySelector('#heroSpeed').textContent,
    browser:document.querySelector('#browserSpeed').textContent,accuracy:document.querySelector('#heroAcc').textContent,
    width:document.documentElement.scrollWidth,viewport:innerWidth}));
  await page.screenshot({path:'results/report_desktop.png',fullPage:true});
  await page.selectOption('#sampleFilter','wrong');
  if(await page.locator('.sample.wrong').count()!==12)throw Error('Incorrect wrong-only filtering');
  await page.locator('.sample').nth(5).click();
  await page.locator('#test').screenshot({path:'results/report_samples.png'});
  await page.selectOption('#curveRun','graph_10epochs');await page.selectOption('#curveAxis','time');
  if(await page.locator('#curve circle').count()!==20)throw Error('10 epoch plot incomplete');
  await page.selectOption('#timingMode','pipeline_seconds_before_export');
  await page.fill('#cfgFilters','8,16');await page.fill('#cfgEpochs','7');
  const command=await page.locator('#command').textContent();
  if(!command.includes('--filters 8,16')||!command.includes('--epochs 7'))throw Error('Command config not applied');
  await page.fill('#cfgFilters','8,$(bad)');
  if(!await page.locator('#copyCommand').isDisabled())throw Error('Invalid command input not rejected');
  await page.fill('#cfgFilters','8,16,32');
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:'results/report_mobile.png',fullPage:true});
  const mobile=await page.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:innerWidth}));
  if(mobile.width>mobile.viewport)throw Error('Mobile horizontal overflow');
  if(errors.length||external.length)throw Error(JSON.stringify({errors,external}));
  const result={passed:true,initial,mobile,errors,external_requests:external,verified:['offline file:// load','desktop/mobile layout','12 random samples','wrong-only filter','sample selection','10-epoch/time-axis curve','timing mode','valid/invalid command settings']};
  fs.writeFileSync('results/report_verification.json',JSON.stringify(result,null,2));console.log(result);
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
