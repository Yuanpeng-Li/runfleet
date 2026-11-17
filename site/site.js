/* RunFleet project page: copy buttons next to the commands. */
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
