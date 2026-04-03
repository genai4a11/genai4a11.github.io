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

      host.innerHTML =
        '<span class="likebtn-wrapper"' +
        ' data-identifier="' + identifier.replace(/"/g, '') + '"' +
        ' data-i18n_like="Clap"' +
        ' data-i18n_unlike="Unclap"' +
        ' data-icon_l="hrt6"' +
        ' data-icon_u="hrt6"' +
        ' data-label_c="true"' +
        ' data-show_like_label="false">' +
        '</span>';
    });

    loadLikeBtnScriptOnce();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountClapWidgets);
  } else {
    mountClapWidgets();
  }
})();
