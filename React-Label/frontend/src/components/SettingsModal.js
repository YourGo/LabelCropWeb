import React from 'react';

function SettingsModal({ settings, onSettingsChange, onClose }) {
  const updateSetting = (key, value) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>⚙️ Processing Settings</h2>

        <div className="setting-group">
          <label>Threshold: {settings.threshold}</label>
          <input
            type="range"
            min="150"
            max="250"
            value={settings.threshold}
            onChange={(e) => updateSetting('threshold', parseInt(e.target.value))}
          />
          <p>Controls detection sensitivity</p>
        </div>

        <div className="setting-group">
          <label>Padding %: {settings.paddingPercent}</label>
          <input
            type="range"
            min="0"
            max="20"
            value={settings.paddingPercent}
            onChange={(e) => updateSetting('paddingPercent', parseInt(e.target.value))}
          />
          <p>Adds extra space around detected label</p>
        </div>

        <div className="setting-group">
          <label>Border Padding %: {settings.borderPaddingPercent}</label>
          <input
            type="range"
            min="0"
            max="30"
            value={settings.borderPaddingPercent}
            onChange={(e) => updateSetting('borderPaddingPercent', parseInt(e.target.value))}
          />
          <p>Expansion when border is detected around label</p>
        </div>

        <div className="setting-group">
          <label>DPI: {settings.dpi}</label>
          <select
            value={settings.dpi}
            onChange={(e) => updateSetting('dpi', parseInt(e.target.value))}
          >
            {[150, 200, 250, 300, 350, 400, 450, 500, 550, 600].map(dpi => (
              <option key={dpi} value={dpi}>{dpi}</option>
            ))}
          </select>
          <p>Image resolution for processing</p>
        </div>

        <div className="setting-group">
          <label>
            <input
              type="checkbox"
              checked={settings.aspectLock}
              onChange={(e) => updateSetting('aspectLock', e.target.checked)}
            />
            Lock 6:4/4:6 Aspect (auto)
          </label>
          <p>Automatically adjust to standard label ratios</p>
        </div>

        <button onClick={onClose} className="primary">✅ Apply Settings</button>
      </div>
    </div>
  );
}

export default SettingsModal;