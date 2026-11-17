/* RunFleet web demo player (markup: index.html, data: runfleet-demo.json, recorded by scripts/web_demo.py).
   Plays screens recorded from RunFleet on a simulated fleet. Every state is one screen, with several frames when it
   is live; its keys and click targets lead to other recorded states. Without the data the page keeps the still
   picture. */
(function () {
  'use strict';

  var root = document.querySelector('.rfd[data-src]');
  if (!root || !window.fetch) return;

  var RUN = 'uvx --from git+https://github.com/Yuanpeng-Li/runfleet runfleet --demo';
  var win = root.querySelector('.rfd__win');
  var body = root.querySelector('.rfd__body');
  var screen = root.querySelector('.rfd__screen');
  var statusEl = root.querySelector('.rfd__status');
  var toastEl = root.querySelector('.rfd__toast');
  var keysEl = root.querySelector('.rfd__keys');
  var fullBtn = root.querySelector('.rfd__full');
  var tryEl = root.querySelector('.rfd__try');
  var poster = showPoster();

  var data, cur, prevClick, lastCell = null, tick = 0, timer = 0, onScreen = true, lastToast = 0, toastTimer = 0;
  var pointer = 'mouse', lastTap = null, hintTimer = 0;
  var rowEls = [], cache = [], hl, buttons = [];
  var chRatio = 0.6, lhRatio = 1.2, ink = null;
  var reduced = window.matchMedia ? matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
  var coarse = window.matchMedia ? matchMedia('(pointer: coarse)') : { matches: false };

  // Textual's names for punctuation keys; letters and digits are named by themselves.
  var PUNCT = {
    '!': 'exclamation_mark', '"': 'quotation_mark', '#': 'number_sign', '$': 'dollar_sign', '%': 'percent_sign',
    '&': 'ampersand', "'": 'apostrophe', '(': 'left_parenthesis', ')': 'right_parenthesis', '*': 'asterisk',
    '+': 'plus', ',': 'comma', '-': 'minus', '.': 'full_stop', '/': 'slash', ':': 'colon', ';': 'semicolon',
    '<': 'less_than_sign', '=': 'equals_sign', '>': 'greater_than_sign', '?': 'question_mark', '@': 'at',
    '[': 'left_square_bracket', '\\': 'backslash', ']': 'right_square_bracket', '^': 'circumflex_accent',
    '_': 'underscore', '`': 'grave_accent', '{': 'left_curly_bracket', '|': 'vertical_line',
    '}': 'right_curly_bracket', '~': 'tilde'
  };
  var NAMED = {
    Tab: 'tab', Enter: 'enter', Escape: 'escape', Esc: 'escape', ' ': 'space', ArrowUp: 'up', ArrowDown: 'down',
    ArrowLeft: 'left', ArrowRight: 'right', Up: 'up', Down: 'down', Left: 'left', Right: 'right', Home: 'home',
    End: 'end', PageUp: 'pageup', PageDown: 'pagedown', Backspace: 'backspace', Delete: 'delete'
  };
  var LABEL = {
    tab: 'Tab', 'shift+tab': 'Shift+Tab', enter: 'Enter', escape: 'Esc', space: 'Space', up: '\u2191',
    down: '\u2193', left: '\u2190', right: '\u2192', home: 'Home', end: 'End', pageup: 'PgUp',
    pagedown: 'PgDn', backspace: 'Bksp', delete: 'Del'
  };
  var CAP = { 'shift+tab': '\u21e7Tab' };    // keycaps that read better shorter than their label
  var SPOKEN = {                              // keycap names for screen readers, where the glyph says little
    up: 'Up arrow', down: 'Down arrow', left: 'Left arrow', right: 'Right arrow', left_square_bracket: 'Left bracket',
    right_square_bracket: 'Right bracket', question_mark: 'Question mark'
  };
  // Keys the screen keeps even when they lead nowhere: Tab, and the ones that would scroll the page.
  var KEEP = /^(shift\+)?(tab|space|up|down|left|right|pageup|pagedown|home|end)$/;
  var HINT = /^((shift\+)?tab|enter|escape)$/;   // the keys the hint line spells out
  var PUNCT_CHAR = {};
  Object.keys(PUNCT).forEach(function (ch) { PUNCT_CHAR[PUNCT[ch]] = LABEL[PUNCT[ch]] = ch; });

  // Cells that fonts get wrong: East Asian wide glyphs take two cells, combining marks none.
  var WIDE = /[\u1100-\u115f\u2e80-\u303e\u3041-\ua4cf\uac00-\ud7a3\uf900-\ufaff\ufe30-\ufe4f\uff00-\uff60\uffe0-\uffe6\u{1f300}-\u{1f64f}\u{1f900}-\u{1f9ff}\u{20000}-\u{3fffd}]/u;
  var ZERO = /[\u0300-\u036f\u200b-\u200f\u20d0-\u20ff\ufe00-\ufe0f]/;

  // ---- loading ----------------------------------------------------------------------------------------------

  // The <noscript> picture doubles as the poster until the data is in, and stays if it never comes.
  function showPoster() {
    var ns = root.querySelector('noscript');
    var tmp = document.createElement('div');
    tmp.innerHTML = ns ? ns.textContent : '';
    var img = tmp.querySelector('img');
    if (img) {
      img.className = 'rfd__poster';
      ns.parentNode.insertBefore(img, ns);
    }
    return img;
  }

  function whenNear(el, fn) {
    if (!('IntersectionObserver' in window)) return fn();
    var io = new IntersectionObserver(function (entries) {
      if (entries.some(function (e) { return e.isIntersecting; })) { io.disconnect(); fn(); }
    }, { rootMargin: '600px 0px' });
    io.observe(el);
  }

  whenNear(root, function () {
    fetch(root.getAttribute('data-src'))
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(start)
      .catch(function (err) { console.warn('RunFleet demo: keeping the still picture:', err); });
  });

  function start(d) {
    if (!d || d.v !== 1 || !(d.cols > 0) || !(d.rows > 0) || !d.states || !d.states[d.start]) {
      throw new Error('unexpected demo data');
    }
    data = d;
    cur = d.start;
    measure();
    addStyles();
    screen.style.setProperty('--cols', d.cols);
    screen.style.color = color(d.fg);
    screen.style.backgroundColor = color(d.bg);
    for (var y = 0; y < d.rows; y++) rowEls.push(screen.appendChild(el('div', 'r')));
    hl = screen.appendChild(el('div', 'rfd__hl'));
    hl.hidden = true;
    buildKeys();
    paint();
    syncKeys();
    statusEl.textContent = state().title || '';

    if (poster) poster.parentNode.removeChild(poster);
    root.querySelector('.rfd__cap').hidden = win.hidden = false;
    keysEl.hidden = !buttons.length;
    fit();
    wire();
    schedule();
  }

  // ---- rendering --------------------------------------------------------------------------------------------

  function el(tag, cls) {
    var n = document.createElement(tag);
    n.className = cls;
    return n;
  }

  function color(c) {
    return /^#[0-9a-f]{6}$/i.test(c) ? c : 'inherit';
  }

  // Per pixel of font size: the width of "0", and where a box-drawing stroke starts and ends around the baseline
  // (ink) next to the font's ascent and descent. Rows get the stroke's height.
  function measure() {
    var probe = root.appendChild(el('span', 'rfd__probe'));
    chRatio = probe.getBoundingClientRect().width / 1000 || 0.6;
    root.removeChild(probe);
    var ctx = document.createElement('canvas').getContext('2d');
    if (!ctx) return;
    ctx.font = '100px ' + getComputedStyle(screen).fontFamily;
    var m = ctx.measureText('\u2502');
    var a = m.actualBoundingBoxAscent / 100, d = m.actualBoundingBoxDescent / 100;
    if (!(a + d > 0.5)) return;
    lhRatio = Math.min(1.35, Math.max(1.1, a + d));
    var A = m.fontBoundingBoxAscent / 100, D = m.fontBoundingBoxDescent / 100;
    ink = A + D > 0 ? [a, d, A, D] : [a, d, a, d];
  }

  // One class per recorded style, and the block elements (U+2580-259F) drawn as backgrounds, so bars and
  // half blocks fill their cell exactly instead of relying on the font's glyphs.
  function addStyles() {
    var css = data.styles.map(function (s, i) {
      var f = s[2] | 0;
      var deco = (f & 4 ? ' underline' : '') + (f & 8 ? ' line-through' : '');
      return '.rfd .s' + i + '{color:' + color(s[0]) + ';background-color:' + color(s[1]) +
        (f & 1 ? ';font-weight:700' : '') + (f & 2 ? ';font-style:italic' : '') +
        (deco ? ';text-decoration:' + deco : '') + '}';
    });
    for (var cp = 0x2580; cp <= 0x259f; cp++) {
      var rects = blockRects(cp);
      if (!rects) continue;
      css.push('.rfd .b' + cp.toString(16) + '{background:' + rects.map(function (r) {
        return 'linear-gradient(currentColor,currentColor) ' + (r[0] ? '100%' : '0') + ' ' + (r[1] ? '100%' : '0') +
          '/' + r[2] * 100 + '% ' + r[3] * 100 + '% no-repeat';
      }).join(',') + '}');
    }
    document.head.appendChild(el('style', '')).textContent = css.join('\n');
  }

  // [right-aligned, bottom-aligned, width, height] as fractions of the cell.
  function blockRects(cp) {
    if (cp === 0x2580) return [[0, 0, 1, 0.5]];
    if (cp <= 0x2588) return [[0, 1, 1, (cp - 0x2580) / 8]];
    if (cp <= 0x258f) return [[0, 0, (0x2590 - cp) / 8, 1]];
    if (cp === 0x2590) return [[1, 0, 0.5, 1]];
    if (cp === 0x2594) return [[0, 0, 1, 0.125]];
    if (cp === 0x2595) return [[1, 0, 0.125, 1]];
    if (cp < 0x2596) return null;                         // shades: the font draws them
    var q = [4, 8, 1, 13, 9, 7, 11, 2, 6, 14][cp - 0x2596]; // quadrants: 1 UL, 2 UR, 4 LL, 8 LR
    return [[0, 0], [1, 0], [0, 1], [1, 1]].filter(function (_, i) { return q & (1 << i); })
      .map(function (p) { return [p[0], p[1], 0.5, 0.5]; });
  }

  // A row as HTML: one span per run with its width in cells; every non-ASCII character in a box of its own.
  function lineHTML(i) {
    if (cache[i] != null) return cache[i];
    var html = '';
    (data.lines[i] || []).forEach(function (seg) {
      var inner = '', n = 0;
      for (var ch of String(seg[0])) {
        var cp = ch.codePointAt(0);
        if (cp < 0x7f) {
          inner += cp < 0x20 ? ' ' : ch === '<' ? '&lt;' : ch === '&' ? '&amp;' : ch === '>' ? '&gt;' : ch;
          n++;
        } else if (ZERO.test(ch)) {
          inner += ch;
        } else if (cp >= 0x2580 && cp <= 0x259f && blockRects(cp)) {
          inner += '<i class="c b' + cp.toString(16) + '"></i>';
          n++;
        } else if (cp >= 0x2500 && cp <= 0x257f) {
          inner += '<i class="c bx">' + ch + '</i>';
          n++;
        } else {
          var w = WIDE.test(ch) ? 2 : 1;
          inner += '<i class="c' + (w === 2 ? ' w2' : '') + '">' + ch + '</i>';
          n += w;
        }
      }
      html += '<span class="s' + (seg[1] | 0) + '" style="width:calc(' + n + '*var(--cw))">' + inner + '</span>';
    });
    return (cache[i] = html);
  }

  function state(id) {
    return data.states[id || cur];
  }

  function paint() {
    var frames = state().frames;
    var f = frames[reduced.matches ? 0 : tick % frames.length];
    for (var y = 0; y < rowEls.length; y++) {
      if (rowEls[y].li !== f[y]) {
        rowEls[y].li = f[y];
        rowEls[y].innerHTML = lineHTML(f[y]);
      }
    }
  }

  // Font size from the width (and, full screen, the height). Cells are sized in px, in whole 1/64 px (the
  // browser's layout unit), so runs of cells never drift off the grid; rows are whole pixels. Box-drawing glyphs
  // are stretched just enough to overshoot their row, which clips them, so vertical lines join without gaps.
  function fit() {
    var cw = body.clientWidth / data.cols;
    if (fullscreenEl() === win) cw = Math.min(cw, body.clientHeight / data.rows / lhRatio * chRatio);
    cw = Math.floor(cw * 64) / 64;
    var fs = cw / chRatio, lh = Math.max(1, Math.round(fs * lhRatio)), k = 1.3;
    if (fullscreenEl() === win) lh = Math.max(1, Math.min(lh, Math.floor(body.clientHeight / data.rows)));   // rows fit
    if (ink) {
      var base = (lh - (ink[2] + ink[3]) * fs) / 2 + ink[2] * fs;     // baseline, from the top of the row
      var up = lh / 2 - (base - ink[0] * fs), down = base + ink[1] * fs - lh / 2;
      if (up > 1 && down > 1) k = Math.max(1, (lh / 2 + 0.75) / Math.min(up, down));
    }
    screen.style.fontSize = fs + 'px';
    screen.style.setProperty('--cw', cw + 'px');
    screen.style.setProperty('--lh', lh + 'px');
    screen.style.setProperty('--bx', k.toFixed(3));
  }

  // Frames cycle only while the demo is on screen, the tab visible and motion welcome.
  function schedule() {
    clearTimeout(timer);
    var st = state();
    if (st.frames.length > 1 && onScreen && !document.hidden && !reduced.matches) {
      timer = setTimeout(function () { tick++; paint(); schedule(); }, Math.max(100, st.ms || 1000));
    }
  }

  function go(id) {
    if (!data.states[id]) return;
    if (id !== cur) {
      cur = id;
      toastEl.hidden = true;                  // a toast is about the screen it came up on
      paint();
      syncKeys();
      statusEl.textContent = state().title || '';
    }
    hover(lastCell);
    schedule();
  }

  // ---- input ------------------------------------------------------------------------------------------------

  function target(names, id) {
    var keys = state(id).keys || {};
    for (var i = 0; i < names.length; i++) {
      if (Object.prototype.hasOwnProperty.call(keys, names[i])) return keys[names[i]];
    }
    return null;
  }

  // KeyboardEvent -> Textual key names (the second one, for punctuation, is the bare character).
  function keyNames(e) {
    if (e.altKey || e.metaKey || e.isComposing) return null;
    var k = e.key, name = NAMED[k];
    if (name) {
      if (name === 'space') return [e.ctrlKey ? 'ctrl+space' : 'space'];
      return [(e.ctrlKey ? 'ctrl+' : '') + (e.shiftKey ? 'shift+' : '') + name];
    }
    if (/^F([1-9]|1[0-2])$/.test(k)) return [(e.ctrlKey ? 'ctrl+' : '') + k.toLowerCase()];
    if (k && Array.from(k).length === 1) {
      if (e.ctrlKey) return /^[a-z]$/i.test(k) ? ['ctrl+' + k.toLowerCase()] : null;
      return PUNCT[k] ? [PUNCT[k], k] : [k];
    }
    return null;
  }

  function onKey(e) {
    var names = keyNames(e);
    if (!names) return;
    var to = target(names);
    var esc = names[0] === 'escape';
    if (to && !(esc && to === cur)) {
      e.preventDefault();
      go(to);
    } else if (esc) {
      screen.blur();
      toast('Keyboard handed back to the page: Tab moves on; click the screen to take it again.', true);
    } else {
      if (KEEP.test(names[0])) e.preventDefault();     // Tab stays in the demo, and the page does not scroll
      if (!e.ctrlKey) toast();
    }
  }

  function cellAt(e) {
    var r = screen.getBoundingClientRect();
    return [Math.floor((e.clientX - r.left) * data.cols / r.width), Math.floor((e.clientY - r.top) * data.rows / r.height)];
  }

  function hit(id, c, kind) {
    var hot = state(id).hot || [];
    for (var i = 0; i < hot.length; i++) {
      var h = hot[i];
      if (c[0] >= h[0] && c[0] < h[0] + h[2] && c[1] >= h[1] && c[1] < h[1] + h[3] && (h[5] || 'click') === kind) {
        return h;
      }
    }
    return null;
  }

  // A key pressed by a keycap or a hint bar item: where it leads, or a toast (Esc with nowhere to go does nothing).
  function press(name) {
    var to = target([name, PUNCT_CHAR[name] || name]);
    if (to) go(to);
    else if (name !== 'escape') toast();
  }

  function click(c) {
    prevClick = cur;
    var h = hit(null, c, 'click'), k = !h && hit(null, c, 'key'), opens = h && hit(null, c, 'dbl');
    clearTimeout(hintTimer);
    if (h && h[4] === cur && opens && opens[4] !== cur) {   // a click on what is selected already only selects
      hintTimer = setTimeout(function () {
        toast((pointer === 'touch' ? 'Double-tap' : 'Double-click') + ' or press Enter to open it.', true);
      }, 450);
    }
    if (h) go(h[4]);
    else if (k) press(k[4]);
  }

  function dblclick(c) {
    clearTimeout(hintTimer);                  // what was double-clicked is on the screen the first click was on
    var h = (prevClick && data.states[prevClick] && hit(prevClick, c, 'dbl')) || hit(null, c, 'dbl');
    if (h) go(h[4]);
  }

  // Outline the click target under the mouse.
  function hover(c) {
    var h = c && (hit(null, c, 'click') || hit(null, c, 'dbl') || hit(null, c, 'key'));
    screen.classList.toggle('is-hot', !!h);
    hl.hidden = !h;
    if (!h) return;
    hl.style.left = h[0] / data.cols * 100 + '%';
    hl.style.top = h[1] / data.rows * 100 + '%';
    hl.style.width = h[2] / data.cols * 100 + '%';
    hl.style.height = h[3] / data.rows * 100 + '%';
  }

  function toast(text, force) {
    var now = Date.now();
    if (!force && now - lastToast < 8000) return;
    lastToast = now;
    toastEl.textContent = '';
    if (text) {
      toastEl.textContent = text;
    } else {
      toastEl.appendChild(document.createTextNode('Not wired in this web demo. Try the real thing: '));
      toastEl.appendChild(el('code', '')).textContent = RUN;
    }
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.hidden = true; }, text ? 3500 : 6000);
  }

  function label(key) {
    if (LABEL[key]) return LABEL[key];
    var m = /^ctrl\+(.+)$/.exec(key);
    return m ? 'Ctrl+' + label(m[1]).toUpperCase() : key;
  }

  // One keycap per legend key; what it does is its title (and, with the key's name, its accessible name).
  function buildKeys() {
    var legend = data.legend || [];
    keysEl.setAttribute('role', 'group');
    keysEl.setAttribute('aria-label', 'Demo keys');
    legend.forEach(function (l) {
      var b = keysEl.appendChild(el('button', 'rfd__key'));
      b.type = 'button';
      b.textContent = CAP[l[0]] || label(l[0]);
      b.title = l[1];
      b.setAttribute('aria-label', (SPOKEN[l[0]] || label(l[0])) + (l[1] === l[0] ? '' : ': ' + l[1]));
      b.addEventListener('click', function () { press(l[0]); });
      buttons.push([b, l[0]]);
    });
    var hint = legend.filter(function (l) { return HINT.test(l[0]); })
      .map(function (l) { return label(l[0]) + ' ' + l[1]; });
    hint.push('Esc at the top level hands the keyboard back to the page');
    var hintEl = root.querySelector('.rfd__hint');
    hintEl.textContent = hint.join(' \u00b7 ');
    if (coarse.matches) {                     // a touch screen: taps, and the keycaps instead of a keyboard
      tryEl.textContent = tryEl.textContent.replace('double-click', 'double-tap').replace('click', 'tap');
      hintEl.hidden = true;
    }
    screen.setAttribute('aria-label', 'RunFleet demo, recorded on a simulated fleet. Click a host, a job or a ' +
      'tab (double-click opens a table row), or use the keys: ' + legend.map(function (l) { return label(l[0]) + ' ' + l[1]; }).join(', ') +
      '. Escape with nothing to go back to leaves the demo.');
  }

  function syncKeys() {
    buttons.forEach(function (b) {
      b[0].setAttribute('aria-disabled', target([b[1], PUNCT_CHAR[b[1]] || b[1]]) ? 'false' : 'true');
    });
  }

  function fullscreenEl() {
    return document.fullscreenElement || document.webkitFullscreenElement;
  }

  function wire() {
    screen.addEventListener('keydown', onKey);
    screen.addEventListener('pointerdown', function (e) { pointer = e.pointerType || 'mouse'; });
    screen.addEventListener('click', function (e) {
      if (e.detail > 1) return;               // the second click of a double click: dblclick handles it
      var c = cellAt(e), now = Date.now();
      if (pointer === 'touch' && lastTap && now - lastTap[0] < 450 &&
          Math.abs(c[0] - lastTap[1][0]) <= 2 && Math.abs(c[1] - lastTap[1][1]) <= 1) {
        lastTap = null;                       // a double tap where the browser counts two single taps
        dblclick(c);
        return;
      }
      lastTap = pointer === 'touch' ? [now, c] : null;
      click(c);
    });
    screen.addEventListener('dblclick', function (e) { dblclick(cellAt(e)); });
    screen.addEventListener('pointermove', function (e) {
      if (e.pointerType === 'mouse') hover(lastCell = cellAt(e));
    });
    screen.addEventListener('pointerleave', function () { hover(lastCell = null); });

    if ('ResizeObserver' in window) new ResizeObserver(fit).observe(body);
    else window.addEventListener('resize', fit);

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        onScreen = entries[entries.length - 1].isIntersecting;
        schedule();
      }).observe(win);
    }
    document.addEventListener('visibilitychange', schedule);
    var onMotion = function () { paint(); schedule(); };
    if (reduced.addEventListener) reduced.addEventListener('change', onMotion);
    else if (reduced.addListener) reduced.addListener(onMotion);

    if (document.fullscreenEnabled || document.webkitFullscreenEnabled) {
      fullBtn.hidden = false;
      fullBtn.addEventListener('click', function () {
        if (fullscreenEl()) {
          (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        } else {
          var p = (win.requestFullscreen || win.webkitRequestFullscreen).call(win);
          if (p && p.catch) p.catch(function () {});
        }
      });
      ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (ev) {
        document.addEventListener(ev, function () {
          var on = fullscreenEl() === win;
          fullBtn.setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen');
          fullBtn.title = fullBtn.getAttribute('aria-label');
          fit();
          if (on) screen.focus();
        });
      });
    }
  }
})();
