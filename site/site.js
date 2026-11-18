/* RunFleet project page: the demo tour, and copy buttons next to the commands. */
(function () {
  'use strict';

  var status = document.getElementById('copy-status');
  var mac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

  // The button says what happened for a moment; the live region says it to screen readers.
  function say(btn, label, text) {
    btn.textContent = label;
    btn.classList.add('is-done');
    if (status) status.textContent = text;
    clearTimeout(btn._t);
    btn._t = setTimeout(function () {
      btn.textContent = 'Copy';
      btn.classList.remove('is-done');
      if (status) status.textContent = '';
    }, 2000);
  }

  // Without the Clipboard API (plain http, old browsers) the command is selected, without its prompt, for a manual copy.
  function select(btn) {
    var code = btn.parentNode.querySelector('code');
    var prompt = code.querySelector('.cmd__prompt');
    var range = document.createRange();
    range.selectNodeContents(code);
    if (prompt) range.setStartAfter(prompt);
    var sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    say(btn, mac ? '⌘C' : 'Ctrl+C', 'Command selected: press ' + (mac ? 'Command+C' : 'Ctrl+C') + ' to copy it');
  }

  // The tour, in the demo window's top bar: each button sends the demo to one recorded screen and stays marked while
  // the demo shows it (any tab of the same job counts; "@…" marks a screen reached another way, which is the same
  // screen). The buttons sit next to the screen they change, so nothing scrolls and the focus stays on the button.
  var player = document.querySelector('.rfd[data-src]');
  var tour = Array.prototype.slice.call(document.querySelectorAll('.tour button[data-go]'));

  function shows(btn, id) {
    var go = btn.getAttribute('data-go');
    id = id.split('@')[0];
    if (/^job\//.test(go)) return id.indexOf(go.replace(/[^/]+$/, '')) === 0;
    return id === go;
  }

  if (player && tour.length) {
    player.addEventListener('rfd:ready', function (e) {
      tour.forEach(function (btn) {
        if (!e.detail.has(btn.getAttribute('data-go'))) btn.parentNode.hidden = true;   // not in this recording
      });
    });
    player.addEventListener('rfd:state', function (e) {
      tour.forEach(function (btn) {
        if (shows(btn, e.detail)) btn.setAttribute('aria-current', 'true');
        else btn.removeAttribute('aria-current');
      });
    });
    tour.forEach(function (btn) {
      btn.addEventListener('click', function () {
        player.dispatchEvent(new CustomEvent('rfd:go', { detail: btn.getAttribute('data-go') }));
      });
    });
  }

  Array.prototype.forEach.call(document.querySelectorAll('button.copy[data-copy]'), function (btn) {
    btn.addEventListener('click', function () {
      var text = btn.getAttribute('data-copy');
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () {
          say(btn, 'Copied', 'Copied to the clipboard');
        }, function () { select(btn); });
      } else {
        select(btn);
      }
    });
  });
})();
