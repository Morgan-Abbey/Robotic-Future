// Browser checks for bci_demo_standalone.html: port fidelity against the Python simulator, screenshots, interactions.
// three.js is served from a local copy (the page loads it from cdnjs): `npm pack three@0.128.0` and untar it.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
// usage: node check_demo.js <three.min.js (r128)> <screenshot dir> [fidelity|shots|interact|all]
const THREE = fs.readFileSync(process.argv[2], 'utf8');
const SP = path.resolve(process.argv[3] || '.');
const URL = 'file://' + path.resolve(__dirname, 'bci_demo_standalone.html');

async function open(browser, opts = {}) {
  const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1280, height: 900 }, colorScheme: opts.scheme || 'light', deviceScaleFactor: opts.dpr || 1, hasTouch: !!opts.touch });
  await ctx.route('https://cdnjs.cloudflare.com/**', (r) => r.fulfill({ status: 200, contentType: 'application/javascript', body: THREE }));
  await ctx.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (['error', 'warning'].includes(m.type())) errors.push(m.type() + ': ' + m.text()); });
  await page.goto(URL);
  await page.waitForTimeout(600);
  return { page, ctx, errors };
}

(async () => {
  const browser = await chromium.launch({ args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const mode = process.argv[4] || 'all';
  if (mode === 'fidelity' || mode === 'all') {
    const { page, errors } = await open(browser);
    const cells = [
      { ctrl: 'direct', acc: 65, budget: '0.25' }, { ctrl: 'proportional', acc: 75, budget: '0.25' }, { ctrl: 'evidence', acc: 75, budget: '0.25' },
      { ctrl: 'shared', acc: 65, budget: '0.25' }, { ctrl: 'evidence', acc: 85, budget: '0.05' }, { ctrl: 'proportional', acc: 85, budget: '0.05' },
      { ctrl: 'shared', acc: 85, budget: '0.25' }, { ctrl: 'proportional', acc: 65, budget: '1.0' },
    ];
    for (const c of cells) {
      const r = await page.evaluate((c) => {
        const S = window.BCISim, n = 300, res = [];
        for (let i = 0; i < n; i++) res.push(S.runReach({ ctrl: c.ctrl, acc: c.acc, budget: c.budget, goalIdx: i % 4, seed: 100000 + i }));
        const idle = []; for (let i = 0; i < 40; i++) idle.push(S.runIdle({ ctrl: c.ctrl, acc: c.acc, budget: c.budget, seed: 200000 + i }));
        const t = res.map((x) => x.time), m = t.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(t.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1));
        return { success: res.filter((x) => x.success).length / n, time: m, ci: 1.96 * sd / Math.sqrt(n), idle: idle.reduce((a, b) => a + b, 0) / idle.length / 0.05 };
      }, c);
      console.log(JSON.stringify(c), '-> success %s time %s ± %s idle %s false steps/min', (100 * r.success).toFixed(0) + '%', r.time.toFixed(1), r.ci.toFixed(1), r.idle.toFixed(1));
    }
    console.log('errors:', errors);
    await page.context().close();
  }
  if (mode === 'shots' || mode === 'all') {
    const out = SP + '/'; fs.mkdirSync(SP, { recursive: true });
    // desktop light, arm watching
    let o = await open(browser);
    await o.page.waitForTimeout(6000);
    console.log('nogl hidden:', await o.page.$eval('#nogl', (e) => e.hidden), 'canvas size', await o.page.$eval('#gl', (e) => [e.width, e.height]));
    await o.page.screenshot({ path: out + 'arm_light_desktop.png' });
    console.log('arm light errors', o.errors);
    // dark
    await o.page.emulateMedia({ colorScheme: 'dark' }); await o.page.waitForTimeout(500);
    await o.page.screenshot({ path: out + 'arm_dark_desktop.png' });
    // cursor 2D
    await o.page.click('#tab-cursor'); await o.page.waitForTimeout(3500);
    await o.page.screenshot({ path: out + 'cursor2d_dark_desktop.png' });
    await o.page.emulateMedia({ colorScheme: 'light' });
    await o.page.click('#seg-axes button[data-v="1"]'); await o.page.waitForTimeout(3000);
    await o.page.screenshot({ path: out + 'cursor1d_light_desktop.png' });
    console.log('desktop errors', o.errors);
    await o.ctx.close();
    // phone
    o = await open(browser, { viewport: { width: 400, height: 860 }, dpr: 2, touch: true });
    await o.page.waitForTimeout(5000);
    await o.page.screenshot({ path: out + 'arm_light_phone.png', fullPage: true });
    const sw = await o.page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    console.log('phone scrollWidth vs innerWidth', sw);
    await o.ctx.close();
  }
  if (mode === 'interact' || mode === 'all') {
    const { page, errors } = await open(browser);
    const st = () => page.evaluate(() => { const S = window.BCISim.state; return { mode: S.mode, view: S.view, q: S.task.q.map((v) => +v.toFixed(2)), j: S.task.j, intent: S.last.intent, hits: S.stats[S.view].hits, trials: S.stats[S.view].trials, n: S.task.n, tol: S.task.cfg.tol }; });
    const setAcc = (v) => page.$eval('#acc', (el, v) => { el.value = v; el.dispatchEvent(new Event('input')); el.dispatchEvent(new Event('change')); }, v);
    const ok = (name, cond, extra) => console.log((cond ? 'PASS ' : 'FAIL ') + name + (extra ? '  ' + JSON.stringify(extra) : ''));
    await setAcc(99);
    ok('starts in watch mode', (await st()).mode === 'watch');
    // keyboard takes over
    await page.keyboard.down('ArrowRight'); await page.waitForTimeout(250);
    ok('key press switches to play mode', (await st()).mode === 'play');
    await page.click('#btn-reset'); await page.waitForTimeout(100);
    const q0 = (await st()).q.slice();
    await page.keyboard.down('ArrowRight'); await page.waitForTimeout(2600); await page.keyboard.up('ArrowRight');
    const s1 = await st();
    ok('holding right moves selected joint 0 positive', s1.q[0] > 0.15 && s1.j === 0, s1);
    await page.keyboard.down('ArrowLeft'); await page.waitForTimeout(2600); await page.keyboard.up('ArrowLeft');
    const s2 = await st();
    ok('holding left moves it back', s2.q[0] < s1.q[0] - 0.1, s2);
    // clench switches joint (jaw detected 90%: try up to 4 times)
    let sw = false; for (let i = 0; i < 4 && !sw; i++) { await page.keyboard.press('Space'); await page.waitForTimeout(700); sw = (await st()).j === 1; }
    ok('Space clench selects next joint', sw, await st());
    // idle: rest intent while nothing held
    await page.waitForTimeout(1200); ok('nothing held means rest intent', (await st()).intent === 0 || (await st()).intent === 3, await st());
    // pointer pad
    const pad = await page.$('#pad-right'); const bb = await pad.boundingBox();
    await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2); await page.mouse.down(); await page.waitForTimeout(500);
    ok('pointer hold on pad sets right intent', (await st()).intent === 2 || (await st()).intent === 3, await st());
    await page.mouse.up(); await page.waitForTimeout(400);
    ok('release returns to rest', (await st()).intent === 0, await st());
    // cursor mode
    await page.click('#tab-cursor'); await page.waitForTimeout(300);
    let s = await st(); ok('cursor tab: 2 axes, tol 0.1', s.view === 'cursor' && s.n === 2 && s.tol === 0.1, s);
    ok('clench pad visible in 2D', await page.$eval('#pad-clench', (e) => !e.hidden));
    await page.click('#seg-axes button[data-v="1"]'); await page.waitForTimeout(300);
    s = await st(); ok('1D cursor: 1 axis, clench pad hidden', s.n === 1 && (await page.$eval('#pad-clench', (e) => e.hidden)), s);
    // play 1D to the goal with a simple script reading state
    await page.click('#btn-reset');
    let reached = false;
    for (let i = 0; i < 160 && !reached; i++) {
      const g = await page.evaluate(() => { const S = window.BCISim.state; return { goal: S.task.goal[0], q: S.task.q[0], hits: S.stats.cursor.hits }; });
      if (g.hits > 0) { reached = true; break; }
      const dir = g.goal - g.q > 0.04 ? 'ArrowRight' : g.goal - g.q < -0.04 ? 'ArrowLeft' : null;
      if (dir) await page.keyboard.down(dir); await page.waitForTimeout(150); if (dir) await page.keyboard.up(dir);
    }
    s = await st(); ok('a human-style loop reaches a 1D cursor target at 99% accuracy', reached, s);
    // soak: watch mode at 4x
    await page.click('#seg-mode button[data-v="watch"]'); await page.click('#tab-arm'); await setAcc(75); await page.click('#seg-speed button[data-v="4"]'); await page.click('#btn-reset');
    await page.waitForTimeout(25000);
    s = await st(); const rest = await page.$eval('#st-rest', (e) => e.textContent), tm = await page.$eval('#st-time', (e) => e.textContent);
    ok('soak: 4x watch completes trials', s.trials >= 3, { trials: s.trials, hits: s.hits, avgTime: tm, rest });
    await page.screenshot({ path: SP + '/arm_after_soak.png' });
    console.log('errors:', errors.filter((e) => !/ERR_FAILED/.test(e)));
    await page.context().close();
  }
  await browser.close();
})();
