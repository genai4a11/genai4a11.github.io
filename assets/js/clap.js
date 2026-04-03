/* LikeBtn clap widget loader
 * Mounts a LikeBtn widget into every element with [data-clap].
 * Keyed per page using window.location.pathname so counts are
 * aggregated per page URL on the LikeBtn backend.
 * The LikeBtn script is only injected once per page load.
 */
(function () {
  function loadLikeBtnScriptOnce() {
    if (document.getElementById('likebtn_wjs')) return;
    var s = document.createElement('script');
    s.id = 'likebtn_wjs';
    s.src = '//w.likebtn.com/js/w/widget.js';
    s.async = true;
    (document.body || document.head).appendChild(s);
  }

  function mountClapWidgets() {
    var hosts = document.querySelectorAll('[data-clap]');
    if (!hosts.length) return;

    var identifier = window.location.pathname;

    hosts.forEach(function (host) {
      if (host.dataset.clapMounted) return;
      host.dataset.clapMounted = '1';

      var span = document.createElement('span');
      span.className = 'likebtn-wrapper';
      span.setAttribute('data-identifier', identifier);
      span.setAttribute('data-i18n_like', 'Clap');
      span.setAttribute('data-i18n_unlike', 'Unclap');
      span.setAttribute('data-icon_l', 'hrt6');
      span.setAttribute('data-icon_u', 'hrt6');
      span.setAttribute('data-label_c', 'true');
      span.setAttribute('data-show_like_label', 'false');

      while (host.firstChild) host.removeChild(host.firstChild);
      host.appendChild(span);
    });

    loadLikeBtnScriptOnce();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountClapWidgets);
  } else {
    mountClapWidgets();
  }
})();
