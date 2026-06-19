(() => {
  'use strict';

  const PLUGIN_ID = kintone.$PLUGIN_ID;
  const fields = ['apiBaseUrl', 'tenantId', 'drawingNoField', 'productNameField', 'pdfFileField'];

  const getElement = (id) => document.getElementById(id);

  const config = kintone.plugin.app.getConfig(PLUGIN_ID);
  fields.forEach((field) => {
    const element = getElement(field);
    if (element) {
      element.value = config[field] || '';
    }
  });

  getElement('save').addEventListener('click', () => {
    const nextConfig = fields.reduce((acc, field) => {
      acc[field] = getElement(field).value.trim();
      return acc;
    }, {});

    if (!nextConfig.apiBaseUrl) {
      window.alert('API Base URLを入力してください。');
      return;
    }

    kintone.plugin.app.setConfig(nextConfig);
  });

  getElement('cancel').addEventListener('click', () => {
    window.location.href = '../../' + kintone.app.getId() + '/plugin/';
  });
})();
