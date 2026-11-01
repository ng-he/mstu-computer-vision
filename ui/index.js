export default function mount(plugin) {
  const el = plugin.el

  /// The dot lights while objects are still being found, not merely while a
  /// model is loaded: a camera can be watching an empty room.
  let found = 0

  plugin.fit(320)

  el('model').addEventListener('change', (event) => plugin.set('model', event.target.value))
  el('browse').addEventListener('click', () => plugin.pick('model', { kind: 'file' }))

  el('classes').addEventListener('change', (event) => plugin.set('classes', event.target.value))

  /// An empty number field means the default rather than NaN.
  const number = (id, field) =>
    el(id).addEventListener('change', (event) => {
      const value = event.target.value === '' ? 0 : Number(event.target.value)

      if (Number.isFinite(value)) plugin.set(field, value)
    })

  number('confidence', 'confidence')
  number('most', 'max_objects')

  el('only').addEventListener('change', (event) => plugin.set('only_when_found', event.target.checked))
  el('preview').addEventListener('change', (event) => plugin.set('preview', event.target.checked))

  plugin.onSettings((settings) => {
    el('model').value = settings.model ?? ''
    el('confidence').value = settings.confidence ?? 0.5
    el('most').value = settings.max_objects || ''
    el('classes').value = settings.classes ?? ''
    el('only').checked = settings.only_when_found ?? false
    el('preview').checked = settings.preview ?? true
  })

  /// One chip per class in view. Rebuilt only when the set changes, so the
  /// chips do not flicker four times a second.
  let showing = null

  const chips = (classes) => {
    if (classes === showing) return

    showing = classes
    el('seen').replaceChildren()

    for (const name of classes ? classes.split(',') : []) {
      const chip = document.createElement('span')

      chip.className = 'mstu-chip'
      chip.textContent = name
      el('seen').append(chip)
    }
  }

  plugin.onLive((live) => {
    const { state, images, objects, dropped, fps, backend, note, objects_now, preview } = live

    chips(objects_now)

    /// The host serves the file the plugin keeps; the count makes the url
    /// change so the picture is fetched again rather than cached.
    if (preview) {
      el('shot').src = `${plugin.mediaUrl(preview)}&image=${images ?? 0}`
    }

    el('shot').hidden = !preview

    el('state').textContent = backend && backend !== 'none' ? `${state} · ${backend}` : state
    el('images').textContent = images ?? 0
    el('objects').textContent = objects ?? 0
    el('fps').textContent = fps ?? 0
    el('dropped').textContent = dropped ? `${dropped} let go` : 'nothing let go'

    el('note').textContent = note ?? ''
    el('note').hidden = !note

    el('dot').classList.toggle('on', state === 'detecting' && objects !== found)

    found = objects ?? 0
  })

  plugin.onState(({ running }) => {
    if (!running) el('dot').classList.remove('on')
  })
}
