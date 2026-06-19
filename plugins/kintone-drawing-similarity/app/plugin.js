(() => {
  'use strict';

  const PLUGIN_ID = kintone.$PLUGIN_ID;

  const normalizeBaseUrl = (value) => String(value || '').replace(/\/+$/, '');

  const getFieldValue = (record, fieldCode) => {
    if (!fieldCode || !record[fieldCode]) {
      return '';
    }
    return record[fieldCode].value || '';
  };

  const setStatus = (panel, message) => {
    const status = panel.querySelector('.pb-similarity-status');
    status.textContent = message;
  };

  const renderResults = (panel, results) => {
    const list = panel.querySelector('.pb-similarity-list');
    list.textContent = '';

    if (!results.length) {
      setStatus(panel, '類似図面は見つかりませんでした。');
      return;
    }

    setStatus(panel, results.length + '件の候補を表示しています。');

    results.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'pb-similarity-item';

      const body = document.createElement('div');
      const link = document.createElement('a');
      link.className = 'pb-similarity-link';
      link.href = '/k/' + kintone.app.getId() + '/show#record=' + encodeURIComponent(item.recordId);
      link.textContent = item.drawingNo || 'record ' + item.recordId;

      const meta = document.createElement('div');
      meta.className = 'pb-similarity-meta';
      meta.textContent = [item.productName, item.customer].filter(Boolean).join(' / ');

      const score = document.createElement('div');
      score.className = 'pb-similarity-score';
      score.textContent = Math.round(Number(item.score || 0) * 100) + '%';

      body.append(link, meta);
      li.append(body, score);
      list.append(li);
    });
  };

  const createPanel = (button) => {
    const panel = document.createElement('section');
    panel.className = 'pb-similarity-panel';
    panel.innerHTML = [
      '<div class="pb-similarity-header">',
      '<h2 class="pb-similarity-title">類似図面検索</h2>',
      '</div>',
      '<div class="pb-similarity-status">検索ボタンを押すと候補を表示します。</div>',
      '<ul class="pb-similarity-list"></ul>'
    ].join('');

    button.parentNode.insertAdjacentElement('afterend', panel);
    return panel;
  };

  kintone.events.on('app.record.detail.show', (event) => {
    if (document.getElementById('pb-similarity-search')) {
      return event;
    }

    const config = kintone.plugin.app.getConfig(PLUGIN_ID);
    const apiBaseUrl = normalizeBaseUrl(config.apiBaseUrl);
    const header = kintone.app.record.getHeaderMenuSpaceElement();
    const button = document.createElement('button');
    button.id = 'pb-similarity-search';
    button.className = 'pb-similarity-button';
    button.type = 'button';
    button.textContent = '類似図面検索';
    header.appendChild(button);

    const panel = createPanel(button);

    button.addEventListener('click', async () => {
      if (!apiBaseUrl) {
        setStatus(panel, 'プラグイン設定でAPI Base URLを設定してください。');
        return;
      }

      button.disabled = true;
      setStatus(panel, '検索しています...');

      try {
        const response = await fetch(apiBaseUrl + '/similar', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            appId: kintone.app.getId(),
            recordId: event.recordId,
            tenantId: config.tenantId || 'default',
            drawingNo: getFieldValue(event.record, config.drawingNoField),
            productName: getFieldValue(event.record, config.productNameField),
            limit: 10
          })
        });

        if (!response.ok) {
          throw new Error('API returned ' + response.status);
        }

        const data = await response.json();
        renderResults(panel, Array.isArray(data.results) ? data.results : []);
      } catch (error) {
        setStatus(panel, '類似図面検索に失敗しました: ' + error.message);
      } finally {
        button.disabled = false;
      }
    });

    return event;
  });
})();
