(function () {
  const D = window.DATA, R = D.results;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const pct = v => (v * 100).toFixed(1) + '%';
  const f3 = v => v.toFixed(3);

  /* ---------- headline stats ---------- */
  const d = R.test_default_threshold, c = R.test_chosen_threshold;
  const stats = [
    [R.n_samples, 'patient samples', ''],
    [pct(d.accuracy), 'test accuracy (threshold 0.5)', 'hl'],
    [f3(R.test_roc_auc), 'ROC-AUC on unseen patients', ''],
    [d.FN + ' → ' + c.FN, 'missed cancers after tuning the threshold', 'hl']
  ];
  $('#stats').innerHTML = stats.map(s => `<div class="stat ${s[2]}"><b>${s[0]}</b><span>${s[1]}</span></div>`).join('');

  /* ---------- model comparison dot plot ---------- */
  const cv = R.cv_comparison;
  const metrics = { f1: 'F1', recall: 'Recall', precision: 'Precision', accuracy: 'Accuracy', roc_auc: 'ROC-AUC' };
  const LO = 0.90, HI = 1.0, pos = v => ((v - LO) / (HI - LO)) * 100;
  let metric = 'f1';
  const seg = $('#metricSeg');
  seg.innerHTML = Object.entries(metrics).map(([k, v]) => `<button data-k="${k}" aria-pressed="${k === metric}">${v}</button>`).join('');
  function drawDots() {
    const rows = [...cv].sort((a, b) => b[metric] - a[metric]);
    $('#dots').innerHTML = rows.map((r, i) => {
      const sd = metric === 'f1' ? r.f1_std : 0;
      const w = sd ? `<i class="whisk" style="left:${pos(Math.max(LO, r.f1 - sd))}%;width:${pos(Math.min(HI, r.f1 + sd)) - pos(Math.max(LO, r.f1 - sd))}%"></i>` : '';
      return `<div class="dotrow ${i === 0 ? 'top' : ''}"><span>${r.Model}</span><div class="track">${w}<i class="dot" style="left:${pos(r[metric])}%" title="${r.Model}: ${f3(r[metric])}"></i></div><b>${f3(r[metric])}</b></div>`;
    }).join('');
    $('#dotNote').textContent = metric === 'f1'
      ? 'Grey lines show ±1 standard deviation across the 5 folds. They overlap heavily: the top models cannot be reliably told apart.'
      : 'Axis runs from 0.90 to 1.00, so small gaps look bigger than they are.';
  }
  seg.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    metric = b.dataset.k; $$('button', seg).forEach(x => x.setAttribute('aria-pressed', x === b)); drawDots();
  });
  drawDots();

  /* ---------- tuning table ---------- */
  const t = R.tuning, best = R.selected_model;
  $('#tuneBody').innerHTML = Object.entries(t).map(([m, v]) =>
    `<tr class="${m === best ? 'best' : ''}"><td>${m}${m === best ? ' ✓' : ''}</td><td>${Object.entries(v.best_params).map(([k, x]) => k + ' = ' + (x === null ? 'None' : x)).join(', ')}</td><td class="num">${f3(v.cv_f1)}</td></tr>`).join('');

  /* ---------- live threshold lab ---------- */
  const P = D.test.prob, Y = D.test.label, npos = Y.reduce((a, b) => a + b, 0);
  const slider = $('#thr');
  function conf(th) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    P.forEach((p, i) => { const pr = p >= th, y = Y[i]; if (pr && y) tp++; else if (pr) fp++; else if (y) fn++; else tn++; });
    return { tp, fp, tn, fn };
  }
  function render() {
    const th = parseFloat(slider.value), k = conf(th);
    const rec = k.tp / (k.tp + k.fn), prec = k.tp + k.fp ? k.tp / (k.tp + k.fp) : 0;
    const acc = (k.tp + k.tn) / P.length, f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
    $('#thrVal').textContent = th.toFixed(2);
    $('#tp').textContent = k.tp; $('#fp').textContent = k.fp; $('#tn').textContent = k.tn; $('#fn').textContent = k.fn;
    $('#mRec').textContent = pct(rec); $('#mPrec').textContent = pct(prec); $('#mAcc').textContent = pct(acc); $('#mF1').textContent = f3(f1);
    const v = $('#verdict');
    v.className = 'callout ' + (k.fn === 0 ? 'good' : 'warn');
    v.innerHTML = `<p><b>${k.tp} of ${npos}</b> cancers caught, <b>${k.fn}</b> missed. <b>${k.fp}</b> of ${P.length - npos} healthy patients flagged for a follow-up they did not need.</p>`;
  }
  slider.addEventListener('input', render);
  $$('[data-th]').forEach(b => b.addEventListener('click', () => {
    slider.value = b.dataset.th === 'chosen' ? D.chosen_threshold : b.dataset.th; render();
  }));
  slider.value = 0.5; render();

  /* ---------- scroll spy ---------- */
  const links = $$('.nav a'), secs = links.map(a => $(a.getAttribute('href')));
  const io = new IntersectionObserver(es => {
    es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id)); });
  }, { rootMargin: '-30% 0px -60% 0px' });
  secs.forEach(s => s && io.observe(s));

  /* ---------- lightbox + copy ---------- */
  const dlg = $('#lightbox');
  $$('.frame').forEach(f => f.addEventListener('click', () => { $('img', dlg).src = $('img', f).src; $('img', dlg).alt = $('img', f).alt; dlg.showModal(); }));
  dlg.addEventListener('click', () => dlg.close());
  $$('.copy').forEach(b => b.addEventListener('click', () => {
    const txt = b.parentElement.innerText.replace(/Copy\s*$/, '').trim();
    (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(() => { b.textContent = 'Copied'; setTimeout(() => b.textContent = 'Copy', 1400); }).catch(() => { });
  }));
})();
