#!/usr/bin/env python3
"""
Detector de Asteroides - Astrometrica Clone (PyQt5)

Versão atualizada:
- registro + subtração por referência + linkagem (tracklets)
- botões de zoom (Zoom +, Zoom -, Reset Zoom) — roda do mouse desativada por padrão
- manter nível de zoom ao alterar cor / modo de visualização
- escolha do esquema de visualização (Normal, Inverter, Alto Contraste, Colorido 'hot' se matplotlib disponível)
- escolha da cor dos marcadores via seletor de cores
- interface totalmente em português

Dependências principais:
- PyQt5
- numpy
- astropy
- pillow (PIL)
- scipy
- imageio

Dependências opcionais:
- scikit-image (melhor registro)
- matplotlib (colormap 'hot')

Ex.: pip install pyqt5 numpy astropy pillow scipy imageio
Se possível: pip install scikit-image matplotlib
"""
import sys
import io
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from scipy.ndimage import median_filter, label, shift as nd_shift, center_of_mass
import imageio

# tentar importar phase_cross_correlation (mais robusto). Se não disponível, usaremos fallback FFT.
try:
    from skimage.registration import phase_cross_correlation  # optional, more robust
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

# tentar importar matplotlib colormap (opcional)
try:
    import matplotlib.cm as cm  # usado para modo 'hot'
    HAS_MATPLOTLIB = True
except Exception:
    cm = None
    HAS_MATPLOTLIB = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QColorDialog, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor

DEFAULT_SAMPLE = r"/mnt/data/o60882g0397o.2292439.ch.2905600.XY41.p10.fits"


# ---------------------- UTILIDADES ----------------------

def asinh_normalize(data, scale_percentile=99.5):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = data - np.min(data)
    if np.max(data) <= 0:
        return np.zeros_like(data, dtype=np.uint8)
    scale = np.percentile(data, scale_percentile)
    if scale <= 0:
        scale = np.max(data)
    stretched = np.arcsinh(data / scale)
    stretched = stretched / np.max(stretched)
    return (stretched * 255).astype(np.uint8)


def fits_to_pil(path_or_bytes, resize_to=(900, 700), scale_percentile=99.5):
    """
    Converte FITS (ou imagem comum) para PIL.Image já reduzida (thumbnail).
    Retorna imagem RGB.
    """
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        if p.suffix.lower() in ('.fits', '.fit'):
            hdul = fits.open(str(p))
            data = None
            for h in hdul:
                if isinstance(h.data, np.ndarray):
                    data = h.data
                    break
            hdul.close()
            if data is None:
                raise ValueError('Nenhum dado de imagem no FITS')
            arr8 = asinh_normalize(data, scale_percentile=scale_percentile)
            pil = Image.fromarray(arr8).convert('RGB')
            pil.thumbnail(resize_to)
            return pil
        else:
            pil = Image.open(str(p)).convert('RGB')
            pil.thumbnail(resize_to)
            return pil
    else:
        # caso seja bytes
        b = path_or_bytes
        try:
            hdul = fits.open(io.BytesIO(b))
            data = None
            for h in hdul:
                if isinstance(h.data, np.ndarray):
                    data = h.data
                    break
            hdul.close()
            if data is not None:
                arr8 = asinh_normalize(data, scale_percentile=scale_percentile)
                pil = Image.fromarray(arr8).convert('RGB')
                pil.thumbnail(resize_to)
                return pil
        except Exception:
            pass
        pil = Image.open(io.BytesIO(b)).convert('RGB')
        pil.thumbnail(resize_to)
        return pil


def pil_to_qpix(pil):
    bio = io.BytesIO()
    pil.save(bio, format='PNG')
    data = bio.getvalue()
    qimg = QImage.fromData(data)
    return QPixmap.fromImage(qimg)


# ---------------------- REGISTRO (skimage ou fallback FFT) ----------------------

def _fft_cross_correlation_shift(ref, target):
    """
    Estima deslocamento (dy, dx) para alinhar target com ref via correlação FFT.
    Retorna deslocamento aproximado (float, float).
    """
    reff = np.asarray(ref, dtype=float)
    targ = np.asarray(target, dtype=float)
    F_ref = np.fft.fftn(reff)
    F_targ = np.fft.fftn(targ)
    R = F_ref * np.conj(F_targ)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.ifftn(R)
    cc = np.abs(cc)
    max_idx = np.unravel_index(np.argmax(cc), cc.shape)
    shifts = np.array(max_idx, dtype=float)
    for i in range(shifts.size):
        if shifts[i] > cc.shape[i] // 2:
            shifts[i] -= cc.shape[i]
    # centroid local 3x3 para sub-pixel
    y0, x0 = max_idx
    ny, nx = cc.shape
    ys = [(y0 - 1) % ny, y0, (y0 + 1) % ny]
    xs = [(x0 - 1) % nx, x0, (x0 + 1) % nx]
    window = cc[np.ix_(ys, xs)]
    total = window.sum() + 1e-12
    cy = (window * (np.array(ys)[:, None])).sum() / total
    cx = (window * (np.array(xs)[None, :])).sum() / total
    if cy > ny / 2:
        cy -= ny
    if cx > nx / 2:
        cx -= nx
    return float(cy), float(cx)


def register_frames(pil_frames, upsample_factor=10):
    """
    Alinha frames ao primeiro. Usa phase_cross_correlation se disponível,
    caso contrário usa fallback FFT.
    Retorna aligned_arrays (float arrays) e shifts list (dy,dx).
    """
    arrays = [np.array(p.convert('L'), dtype=float) for p in pil_frames]
    ref = arrays[0]
    aligned = []
    shifts = []
    if HAS_SKIMAGE:
        for arr in arrays:
            try:
                shift, error, diffphase = phase_cross_correlation(ref, arr, upsample_factor=upsample_factor)
                arr_shifted = nd_shift(arr, shift, order=1, mode='reflect')
                aligned.append(arr_shifted)
                shifts.append(shift)
            except Exception:
                aligned.append(arr)
                shifts.append((0.0, 0.0))
    else:
        for arr in arrays:
            try:
                shift = _fft_cross_correlation_shift(ref, arr)
                arr_shifted = nd_shift(arr, shift, order=1, mode='reflect')
                aligned.append(arr_shifted)
                shifts.append(shift)
            except Exception:
                aligned.append(arr)
                shifts.append((0.0, 0.0))
    return aligned, shifts


# ---------------------- DETECÇÃO E LINKAGEM ----------------------

def detect_in_differences(aligned_arrays, n_sigma=5.0, min_area=4):
    """
    Gera referência por mediana, subtrai e detecta componentes em cada diferença.
    Retorna detecções por frame, referência e diffs.
    """
    ref = np.median(np.stack(aligned_arrays, axis=0), axis=0)
    diffs = [arr - ref for arr in aligned_arrays]
    absdiffs = [np.abs(d) for d in diffs]

    dets_per_frame = []
    for d in absdiffs:
        _, med, std = sigma_clipped_stats(d, sigma=3.0)
        thresh_val = med + n_sigma * std
        mask = d > thresh_val
        lbls, n = label(mask)
        dets = []
        for lab in range(1, n + 1):
            ys, xs = np.where(lbls == lab)
            if xs.size < min_area:
                continue
            m = (lbls == lab)
            cy, cx = center_of_mass(d * m)
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            flux = float(d[m].sum())
            dets.append({'x': float(cx), 'y': float(cy), 'area': int(xs.size), 'flux': flux})
        dets_per_frame.append(dets)
    return dets_per_frame, ref, diffs


def link_tracklets(dets_per_frame, max_disp_px=8.0, min_detections=3, linear_resid_thresh=2.5):
    """
    Link simplificado em sequência. Retorna lista de tracklets.
    """
    n_frames = len(dets_per_frame)
    tracklets = []
    if n_frames == 0:
        return []

    for d0 in dets_per_frame[0]:
        tracks = [[(0, d0)]]
        for f in range(1, n_frames):
            new_tracks = []
            for tr in tracks:
                lx, ly = tr[-1][1]['x'], tr[-1][1]['y']
                for d in dets_per_frame[f]:
                    if np.hypot(d['x'] - lx, d['y'] - ly) <= max_disp_px:
                        new_tracks.append(tr + [(f, d)])
            tracks = new_tracks
            if not tracks:
                break
        for tr in tracks:
            if len(tr) >= min_detections:
                ts = np.array([t for t, _ in tr], dtype=float)
                xs = np.array([d['x'] for _, d in tr], dtype=float)
                ys = np.array([d['y'] for _, d in tr], dtype=float)
                px = np.polyfit(ts, xs, 1)
                py = np.polyfit(ts, ys, 1)
                xs_fit = np.polyval(px, ts)
                ys_fit = np.polyval(py, ts)
                resid = np.sqrt(np.mean((xs - xs_fit) ** 2 + (ys - ys_fit) ** 2))
                if resid <= linear_resid_thresh:
                    tracklets.append({'points': tr, 'vx': float(px[0]), 'vy': float(py[0]), 'res': float(resid)})
    # remover duplicatas
    unique = []
    seen = set()
    for t in tracklets:
        sig = tuple(sorted((f, round(p['x'], 1), round(p['y'], 1)) for f, p in t['points']))
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(t)
    return unique


# ---------------------- VIEWER (com Zoom por botões) ----------------------

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pix_item = None
        self.current_scale = 1.0
        self.scale_step = 1.25
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        # desativa zoom pela roda por padrão
        self.wheel_zoom_enabled = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_pixmap(self, pix: QPixmap, reset_zoom: bool = False):
        """
        Define o pixmap mostrado.
        Se reset_zoom=True, zera a transformação (scale=1.0).
        Caso contrário, mantém o nível de zoom atual.
        """
        # guarda escala atual se for manter zoom
        prev_scale = self.current_scale
        self.scene.clear()
        self.pix_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self.pix_item)
        if self.pix_item:
            self.setSceneRect(self.pix_item.boundingRect())
        if reset_zoom:
            self.reset_zoom()
        else:
            # aplicar escala anterior (redefine transform e aplica scale)
            self.resetTransform()
            self.current_scale = prev_scale
            self.scale(self.current_scale, self.current_scale)

    def set_scale(self, scale: float):
        if scale <= 0:
            return
        self.resetTransform()
        self.current_scale = scale
        self.scale(scale, scale)

    def zoom_in(self):
        self.set_scale(self.current_scale * self.scale_step)

    def zoom_out(self):
        self.set_scale(self.current_scale / self.scale_step)

    def reset_zoom(self):
        self.set_scale(1.0)

    def wheelEvent(self, event):
        # ignorar roda para evitar zoom acidental (a menos que explicitamente ativado)
        if self.wheel_zoom_enabled:
            super().wheelEvent(event)
        else:
            event.ignore()


# ---------------------- MAIN WINDOW ----------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Detector de Asteroides - Astrometrica Clone')
        self.resize(1200, 800)

        # dados / estado
        self.raw_paths = []
        self.pil_frames = []
        self.marked_frames = []
        self.qpix_frames = []
        self.current = 0
        self.tracklets = []
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.next_frame)

        # opções visuais
        self.view_color_mode = 'normal'  # normal, invert, contrast, hot (hot só se matplotlib disponível)
        self.marker_qcolor = QColor(255, 0, 0)  # cor padrão dos marcadores (vermelho)

        # construir UI
        self._create_menu()
        self._create_toolbar()
        self._create_main()

    # ---------------- MENU ----------------
    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Arquivo')
        help_menu = menubar.addMenu('Ajuda')

        open_act = QAction('Abrir imagens...', self)
        open_act.triggered.connect(self.load_images)
        file_menu.addAction(open_act)

        open_sample = QAction('Abrir Exemplo (FITS)', self)
        open_sample.triggered.connect(self.load_sample)
        file_menu.addAction(open_sample)

        exit_act = QAction('Sair', self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        about_act = QAction('Sobre', self)
        about_act.triggered.connect(lambda: QMessageBox.information(self, 'Sobre', 'Detector de Asteroides — demo'))
        help_menu.addAction(about_act)

    # ---------------- TOOLBAR ----------------
    def _create_toolbar(self):
        toolbar = self.addToolBar('Principal')
        self.btn_prev = QPushButton('◀'); self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_play = QPushButton('Reproduzir'); self.btn_play.clicked.connect(self.play)
        self.btn_pause = QPushButton('Pausar'); self.btn_pause.clicked.connect(self.pause)
        self.btn_next = QPushButton('▶'); self.btn_next.clicked.connect(self.next_frame)
        self.scan_btn = QPushButton('Varredura'); self.scan_btn.clicked.connect(self.scan)
        self.save_gif_btn = QPushButton('Salvar GIF'); self.save_gif_btn.clicked.connect(self.save_gif)

        # botões de zoom explícitos
        self.zoom_in_btn = QPushButton('Zoom +'); self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn = QPushButton('Zoom -'); self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.reset_zoom_btn = QPushButton('Reset Zoom'); self.reset_zoom_btn.clicked.connect(self._reset_zoom)

        for w in [self.btn_prev, self.btn_play, self.btn_pause, self.btn_next,
                  self.scan_btn, self.save_gif_btn,
                  self.zoom_in_btn, self.zoom_out_btn, self.reset_zoom_btn]:
            toolbar.addWidget(w)

        # combo para escolher modo de visualização
        self.mode_combo = QComboBox()
        modes = ['Normal', 'Inverter', 'Alto Contraste']
        if HAS_MATPLOTLIB:
            modes.append('Colorido (hot)')
        self.mode_combo.addItems(modes)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_combo)

        # botão para escolher cor do marcador
        self.choose_color_btn = QPushButton('Cor do Marcador')
        self.choose_color_btn.clicked.connect(self._choose_marker_color)
        toolbar.addWidget(self.choose_color_btn)

    # ---------------- LAYOUT ----------------
    def _create_main(self):
        central = QWidget()
        self.setCentralWidget(central)
        mainlay = QHBoxLayout(central)

        left_panel = QVBoxLayout()
        self.viewer = ImageViewer()
        left_panel.addWidget(self.viewer)

        ctrl = QHBoxLayout()
        self.lbl_index = QLabel('Imagem: - / -')
        ctrl.addWidget(self.lbl_index)
        ctrl.addStretch()
        ctrl.addWidget(QLabel('Brilho'))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(50)
        self.slider.setMaximum(100)
        self.slider.setValue(99)
        self.slider.valueChanged.connect(self.reload_with_slider)

        ctrl.addWidget(self.slider)
        left_panel.addLayout(ctrl)

        mainlay.addLayout(left_panel)
        central.setLayout(mainlay)

    # -------------------- Funções de carregamento --------------------

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Abrir imagens ou FITS", ".",
                                                "Imagens e FITS (*.fits *.fit *.png *.jpg *.jpeg *.tif *.tiff);;Todos os arquivos (*.*)")
        if not files:
            return
        self.raw_paths = files
        try:
            self._load_raw_to_frames()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar imagens:\n{e}")
            return
        self.current = 0
        self.tracklets = []
        self.marked_frames = list(self.pil_frames)
        # converter e aplicar view mode; ao carregar imagens, resetamos o zoom
        self.qpix_frames = [pil_to_qpix(self._apply_view_mode(p)) for p in self.marked_frames]
        self.viewer.reset_zoom()
        self.show_frame(0, reset_zoom=False)

    def load_sample(self):
        if not Path(DEFAULT_SAMPLE).exists():
            QMessageBox.warning(self, "Atenção", f"Arquivo de exemplo não encontrado:\n{DEFAULT_SAMPLE}")
            return
        self.raw_paths = [DEFAULT_SAMPLE] * 4
        try:
            self._load_raw_to_frames()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar exemplo:\n{e}")
            return
        self.current = 0
        self.tracklets = []
        self.marked_frames = list(self.pil_frames)
        self.qpix_frames = [pil_to_qpix(self._apply_view_mode(p)) for p in self.marked_frames]
        self.viewer.reset_zoom()
        self.show_frame(0, reset_zoom=False)

    def _load_raw_to_frames(self):
        if not self.raw_paths:
            raise ValueError("Nenhum arquivo carregado")
        sp = self.slider.value()
        new_frames = []
        for p in self.raw_paths:
            pil = fits_to_pil(p, resize_to=(900, 700), scale_percentile=sp)
            new_frames.append(pil)
        self.pil_frames = new_frames

    # -------------------- Navegação / playback --------------------

    def show_frame(self, idx, reset_zoom: bool = False):
        """
        Exibe o frame idx. reset_zoom controla se deve redefinir o zoom.
        Ao mudar cor/modo, chamamos show_frame sem reset (mantém zoom).
        Ao carregar imagens novas, chamamos com reset_zoom=False mas explicitamente resetamos antes.
        """
        if not self.qpix_frames:
            self.viewer.scene.clear()
            self.lbl_index.setText("Imagem: - / -")
            return
        idx = max(0, min(idx, len(self.qpix_frames) - 1))
        self.current = idx
        pix = self.qpix_frames[idx]
        # não resetar zoom automaticamente aqui: delegamos a reset quando necessário em load_images/load_sample
        self.viewer.set_pixmap(pix, reset_zoom=reset_zoom)
        self.lbl_index.setText(f"Imagem: {idx + 1} / {len(self.qpix_frames)}")

    def prev_frame(self):
        if not self.qpix_frames:
            return
        self.current = (self.current - 1) % len(self.qpix_frames)
        self.show_frame(self.current, reset_zoom=False)

    def next_frame(self):
        if not self.qpix_frames:
            return
        self.current = (self.current + 1) % len(self.qpix_frames)
        self.show_frame(self.current, reset_zoom=False)

    def play(self):
        if not self.qpix_frames:
            return
        if not self.timer.isActive():
            self.timer.start()

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()

    # -------------------- Zoom handlers (conectados aos botões) --------------------

    def _zoom_in(self):
        self.viewer.zoom_in()

    def _zoom_out(self):
        self.viewer.zoom_out()

    def _reset_zoom(self):
        self.viewer.reset_zoom()

    # -------------------- Processamento / Varredura --------------------

    def scan(self):
        """
        Fluxo:
        1) registra frames
        2) subtrai referência (mediana)
        3) detecta fontes em cada diff
        4) linka em tracklets consistentes
        5) desenha apenas tracklets usando cor escolhida
        """
        if not self.pil_frames or len(self.pil_frames) < 2:
            QMessageBox.information(self, "Informação", "Carregue pelo menos 2 frames para varredura.")
            return

        try:
            # parâmetros ajustáveis
            n_sigma = 5.0
            min_area = 4
            max_disp_px = 8.0
            min_detections = 3
            linear_resid_thresh = 2.5

            aligned, shifts = register_frames(self.pil_frames, upsample_factor=10)
            dets_per_frame, ref, diffs = detect_in_differences(aligned, n_sigma=n_sigma, min_area=min_area)
            tracklets = link_tracklets(dets_per_frame, max_disp_px=max_disp_px, min_detections=min_detections, linear_resid_thresh=linear_resid_thresh)
            self.tracklets = tracklets

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro na detecção:\n{e}")
            return

        # Desenha marcadores sobre cópias das pil_frames apenas para tracklets
        marked = [p.copy() for p in self.pil_frames]
        rgb = (self.marker_qcolor.red(), self.marker_qcolor.green(), self.marker_qcolor.blue())
        for tr in self.tracklets:
            for fidx, det in tr['points']:
                im = marked[fidx]
                draw = ImageDraw.Draw(im)
                x, y = det['x'], det['y']
                r = max(4, int(np.sqrt(det.get('area', 16))))
                bbox = (x - r, y - r, x + r, y + r)
                draw.ellipse(bbox, outline=rgb, width=2)
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=rgb)
            # linha indicativa no primeiro frame
            f0, d0 = tr['points'][0]
            fN, dN = tr['points'][-1]
            im = marked[0]
            draw = ImageDraw.Draw(im)
            draw.line((d0['x'], d0['y'], dN['x'], dN['y']), fill=(0, 255, 0), width=1)

        self.marked_frames = marked
        # converter aplicando modo visual selecionado; mantém zoom atual (não reseta)
        self.qpix_frames = [pil_to_qpix(self._apply_view_mode(p)) for p in self.marked_frames]
        self.show_frame(0, reset_zoom=False)
        QMessageBox.information(self, "Varredura", f"{len(self.tracklets)} tracklet(s) detectado(s).")

    # -------------------- Slider / reprocessamento --------------------

    def reload_with_slider(self):
        if not self.raw_paths:
            return
        try:
            self._load_raw_to_frames()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao reprocessar imagens:\n{e}")
            return
        self.marked_frames = list(self.pil_frames)
        self.qpix_frames = [pil_to_qpix(self._apply_view_mode(p)) for p in self.marked_frames]
        self.tracklets = []
        self.current = 0
        self.show_frame(0, reset_zoom=False)

    # -------------------- Export --------------------

    def save_gif(self):
        if not self.marked_frames:
            QMessageBox.information(self, "Salvar GIF", "Nenhuma imagem a salvar. Faça uma varredura ou carregue imagens.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Salvar GIF", "asteroides.gif", "Arquivos GIF (*.gif)")
        if not fname:
            return
        try:
            arrs = [np.asarray(im.convert('RGB')) for im in self.marked_frames]
            imageio.mimsave(fname, arrs, duration=0.35)
            QMessageBox.information(self, "Salvar GIF", f"GIF salvo em: {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao salvar GIF:\n{e}")

    # -------------------- Visual mode & color picker --------------------

    def _on_mode_changed(self, idx):
        opts = ['normal', 'invert', 'contrast']
        if HAS_MATPLOTLIB:
            opts.append('hot')
        self.view_color_mode = opts[idx] if idx < len(opts) else 'normal'
        # reaplicar visual mode sobre as marked_frames (ou pil_frames se nada marcado)
        base = self.marked_frames if self.marked_frames else self.pil_frames
        if base:
            # não resetar zoom ao apenas mudar cor/modo — manter a escala atual
            self.qpix_frames = [pil_to_qpix(self._apply_view_mode(p)) for p in base]
            self.show_frame(self.current, reset_zoom=False)

    def _choose_marker_color(self):
        col = QColorDialog.getColor(self.marker_qcolor, self, "Escolher cor dos marcadores")
        if col.isValid():
            self.marker_qcolor = col
            # redesenhar tracklets com nova cor (se já existirem)
            if self.tracklets and self.pil_frames:
                marked = [p.copy() for p in self.pil_frames]
                rgb = (self.marker_qcolor.red(), self.marker_qcolor.green(), self.marker_qcolor.blue())
                for tr in self.tracklets:
                    for fidx, det in tr['points']:
                        im = marked[fidx]
                        draw = ImageDraw.Draw(im)
                        x, y = det['x'], det['y']
                        r = max(4, int(np.sqrt(det.get('area', 16))))
                        bbox = (x - r, y - r, x + r, y + r)
                        draw.ellipse(bbox, outline=rgb, width=2)
                        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=rgb)
                    f0, d0 = tr['points'][0]
                    fN, dN = tr['points'][-1]
                    im = marked[0]
                    draw = ImageDraw.Draw(im)
                    draw.line((d0['x'], d0['y'], dN['x'], dN['y']), fill=(0, 255, 0), width=1)
                self.marked_frames = marked
                self.qpix_frames = [pil_to_qpix(self._apply_view_mode(p)) for p in self.marked_frames]
                # manter zoom atual
                self.show_frame(self.current, reset_zoom=False)

    def _apply_view_mode(self, pil_img):
        """
        Aplica esquema de visualização à PIL.Image e retorna nova PIL.Image RGB.
        Modos suportados: normal, invert, contrast, hot (se matplotlib disponível)
        """
        mode = getattr(self, 'view_color_mode', 'normal')
        if mode == 'normal' or pil_img is None:
            return pil_img
        if mode == 'invert':
            try:
                return ImageOps.invert(pil_img.convert('RGB'))
            except Exception:
                return pil_img
        if mode == 'contrast':
            im = pil_img.convert('L')
            enhancer = ImageEnhance.Contrast(im)
            im2 = enhancer.enhance(1.8)
            enhancer_b = ImageEnhance.Brightness(im2)
            im3 = enhancer_b.enhance(1.05)
            return im3.convert('RGB')
        if mode == 'hot' and HAS_MATPLOTLIB:
            arr = np.array(pil_img.convert('L'), dtype=float)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
            cmap = cm.get_cmap('hot')
            colored = (cmap(arr)[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored)
        return pil_img


# ---------------------- RODA APLICATIVO ----------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())