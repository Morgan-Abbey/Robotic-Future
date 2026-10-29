// End-to-end check of the live stack: start the bridge first (see README), then
//   node e2e_check.js <three.min.js r128> <http://127.0.0.1:8765/> <arm log file> <screenshot dir>
const { chromium } = require('playwright');
const fs = require('fs');
const [THREE_PATH, URL, ARMLOG, OUT] = process.argv.slice(2);
const THREE = fs.readFileSync(THREE_PATH, 'utf8');
fs.mkdirSync(OUT, { recursive: true });
const ok = (name, cond, extra) => { console.log((cond ? 'PASS ' : 'FAIL ') + name + (extra !== undefined ? '  ' + JSON.stringify(extra) : '')); if (!cond) process.exitCode = 1; };
const lines = () => (fs.existsSync(ARMLOG) ? fs.readFileSync(ARMLOG, 'utf8').split('\n').filter(Boolean) : []);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch({ args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
  await ctx.route('https://cdnjs.cloudflare.com/**', (r) => r.fulfill({ status: 200, contentType: 'application/javascript', body: THREE }));
  await ctx.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/ERR_FAILED/.test(m.text())) errors.push(m.text()); });
  await page.goto(URL);
  await page.waitForFunction(() => document.getElementById('chip-conn').textContent.includes('mock'), null, { timeout: 8000 });
  const vis = (id) => page.$eval('#' + id, (e) => !e.hidden && getComputedStyle(e).display !== 'none');
  ok('opens on the Live tab, connected to the mock engine', await page.$eval('#tab-live', (e) => e.getAttribute('aria-selected') === 'true'));
  const simOnlyHidden = await page.$eval('#seg-mode', (e) => e.closest('.field').hidden);
  ok('training and arm cards visible, simulator-only controls hidden', (await vis('card-train')) && (await vis('card-arm')) && !(await vis('card-stats')) && simOnlyHidden);
  ok('arm starts in dry run with Enable available', (await page.textContent('#arm-state')).includes('dry run') && (await vis('btn-arm-enable')));
  await sleep(600);
  await page.screenshot({ path: OUT + '/live_top.png' });
  ok('floating E-STOP is always visible in live mode', await vis('estop-float'));
  ok('decoder bars are live', (await page.$$eval('#bars .num', (n) => n.map((e) => Number(e.textContent)))).some((v) => v > 0));

  // --- enabling takes two deliberate clicks
  await page.click('#btn-arm-enable');
  ok('first click only asks for confirmation', (await vis('btn-arm-confirm')) && (await page.textContent('#arm-state')).includes('dry run'));
  await page.click('#btn-arm-confirm');
  await page.waitForFunction(() => document.getElementById('arm-state').textContent.includes('ARMED'), null, { timeout: 4000 });
  ok('second click arms the output', true);

  // --- driving with the keyboard moves the commanded pose and writes lines to the arm sink
  await page.click('#interp').catch(() => {});
  await page.selectOption('#interp', 'direct');
  await sleep(300);
  const before = lines().length;
  await page.keyboard.down('ArrowRight'); await sleep(2500); await page.keyboard.up('ArrowRight');
  const q = await page.evaluate(() => window.BCISim.state.task.q.map((v) => +v.toFixed(2)));
  ok('holding right moves joint 1 (commanded pose)', q[0] > 0.05, q);
  const sent = lines().slice(before);
  ok('motion lines reached the arm sink', sent.length > 0 && sent.some((l) => Number(l.split(' ')[0]) > 0), sent.slice(0, 3));
  await page.screenshot({ path: OUT + '/live_armed.png' });

  // --- e-stop with Esc
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => document.getElementById('arm-state').textContent.includes('E-STOP'), null, { timeout: 3000 });
  const n = lines().length;
  await page.keyboard.down('ArrowRight'); await sleep(1200); await page.keyboard.up('ArrowRight');
  ok('after Esc nothing more is written, even while driving', lines().length === n);
  ok('reset button offered, enable blocked until reset', (await vis('btn-arm-reset')) && (await page.$eval('#btn-arm-enable', (e) => e.disabled || e.hidden)));
  await page.click('#btn-arm-reset');
  await sleep(300);
  ok('reset clears the latch but leaves the arm disabled', (await page.textContent('#arm-state')).includes('dry run'));

  // --- jog needs the arm enabled; zero is available
  await page.click('#btn-zero');

  // --- training cue flow (mock engine runs a short demo session)
  await page.click('#btn-record');
  await page.waitForFunction(() => !document.getElementById('cue').hidden, null, { timeout: 6000 });
  const cue = await page.textContent('#cue-main');
  ok('a cue appears while recording', /HAND/.test(cue), cue);
  await page.screenshot({ path: OUT + '/live_cue.png' });
  await page.waitForFunction(() => document.getElementById('log').textContent.includes('model trained'), null, { timeout: 20000 });
  ok('training finishes and reports a model', true, (await page.textContent('#log')).split('\n').slice(-2));

  // --- the page going away disables the arm (heartbeat)
  await page.click('#btn-arm-enable'); await page.click('#btn-arm-confirm');
  await page.waitForFunction(() => document.getElementById('arm-state').textContent.includes('ARMED'), null, { timeout: 4000 });
  await page.close();
  await sleep(4500);
  const p2 = await ctx.newPage();
  await p2.goto(URL);
  await p2.waitForFunction(() => document.getElementById('chip-conn').textContent.includes('mock'), null, { timeout: 8000 });
  await sleep(500);
  ok('a closed browser tab leaves the arm disabled', (await p2.textContent('#arm-state')).includes('dry run'), await p2.textContent('#arm-state'));
  await p2.screenshot({ path: OUT + '/live_after.png' });
  // --- phone width: no horizontal scroll, floating E-STOP present
  const pctx = await browser.newContext({ viewport: { width: 400, height: 860 }, deviceScaleFactor: 2, hasTouch: true });
  await pctx.route('https://cdnjs.cloudflare.com/**', (r) => r.fulfill({ status: 200, contentType: 'application/javascript', body: THREE }));
  await pctx.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  const p3 = await pctx.newPage();
  await p3.goto(URL);
  await p3.waitForFunction(() => document.getElementById('chip-conn').textContent.includes('mock'), null, { timeout: 8000 });
  await sleep(800);
  const sw = await p3.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  ok('phone width: no horizontal scroll', sw[0] <= sw[1], sw);
  ok('phone width: floating E-STOP visible', await p3.$eval('#estop-float', (e) => !e.hidden));
  await p3.screenshot({ path: OUT + '/live_phone.png', fullPage: true });
  console.log('page errors:', errors);
  if (errors.length) process.exitCode = 1;
  await browser.close();
})();
