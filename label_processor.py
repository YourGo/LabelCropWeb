import fitz
import cv2
import numpy as np
from PIL import Image
try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    zbar_decode = None
    PYZBAR_AVAILABLE = False


class PDFLabelProcessor:
    def __init__(self, threshold=200, padding_percent=2, dpi=300, aspect_lock=True):
        self.threshold = threshold
        self.padding_percent = padding_percent
        self.dpi = dpi
        self.aspect_lock = aspect_lock
        self.cropped_img = None
        self.original_img = None

    def find_label_border(self, img):
        """Detect a thin rectangular border surrounding the label.
        Returns (x,y,w,h) if found, else None.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        H, W = img.shape[:2]
        img_area = float(H * W)
        best = None
        best_area = 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            area = float(w * h)
            area_ratio = area / img_area
            if area_ratio < 0.15 or area_ratio > 0.92:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.4 or aspect > 2.6:
                continue
            band = max(1, int(round(min(w, h) * 0.004)))
            left = edges[y:y+h, max(x-1,0):min(x+band, W)]
            right = edges[y:y+h, max(x+w-band,0):min(x+w+1, W)]
            top = edges[max(y-1,0):min(y+band, H), x:x+w]
            bottom = edges[max(y+h-band,0):min(y+h+1, H), x:x+w]
            edge_score = (np.mean(left) + np.mean(right) + np.mean(top) + np.mean(bottom)) / 255.0
            if edge_score < 0.05:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)
        return best

    def detect_cut_line_y(self, img):
        """Detect a horizontal cut/dashed line and return its y coordinate.
        Returns None if not found. Targets long near-horizontal lines.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        H, W = gray.shape[:2]
        kx = max(25, int(round(W * 0.06)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
        closed = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.Canny(closed, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                                minLineLength=int(W * 0.4), maxLineGap=int(W * 0.02))
        if lines is None:
            return None
        best_len = 0
        best_y = None
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            if abs(y1 - y2) > 4:
                continue
            length = abs(x2 - x1)
            yc = (y1 + y2) // 2
            if yc < int(0.2 * H) or yc > int(0.75 * H):
                continue
            if length > best_len:
                best_len = length
                best_y = yc
        return best_y

    def _align_rect_to_vertical_edges(self, img, rect):
        """Adjust the top of the rect downward if the top band has weak
        vertical-edge energy (likely header text), keeping height fixed.
        Returns (x,y,w,h). Safe no-op when signals are weak.
        """
        x, y, w, h = rect
        H, W = img.shape[:2]
        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            return rect
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        vmag = np.abs(sx)
        row_profile = vmag.mean(axis=1)
        if row_profile.size < 10:
            return rect
        row_profile = cv2.GaussianBlur(row_profile.astype(np.float32), (1, 9), 0).reshape(-1)
        p95 = float(np.percentile(row_profile, 95))
        if p95 <= 1e-3:
            return rect
        thr = max(1e-3, 0.18 * p95)
        min_len = max(2, int(round(0.03 * h)))
        above = row_profile >= thr
        idx = 0
        shift = 0
        while idx < len(above):
            if above[idx]:
                j = idx
                while j < len(above) and above[j]:
                    j += 1
                if (j - idx) >= min_len:
                    shift = idx
                    break
                idx = j
            else:
                idx += 1
        if shift == 0:
            return rect
        max_shift = max(1, int(round(0.06 * h)))
        shift = min(shift, max_shift)
        ny = min(max(0, y + shift), max(0, H - h))
        relax = max(0, int(round(0.02 * h)))
        ny = max(0, ny - relax)
        return x, ny, w, h

    def _anchor_rect_to_barcodes(self, img, rect, margin_frac=0.08):
        """Ensure the rect stays around decoded barcodes: do not place the
        bottom above the barcodes. Only enforces bottom coverage and does not
        push the top downward (to avoid trimming label header). Returns
        (x,y,w,h). No-op if no barcodes found.
        """
        x, y, w, h = rect
        H, W = img.shape[:2]
        brect = self.find_largest_barcode_rect(img, None)
        if brect is None:
            return rect
        bx, by, bw, bh = brect
        bot_anchor = min(H, by + bh + int(round(margin_frac * h)))
        ny = y
        if ny + h < bot_anchor:
            ny = min(max(0, bot_anchor - h), max(0, H - h))
        return x, ny, w, h

    def find_label_region(self, img):
        """Find the approximate region where the actual shipping label is"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.convertScaleAbs(np.abs(sobelx) + 0.5 * np.abs(sobely))

        edge_norm = np.empty_like(edge_mag)
        cv2.normalize(edge_mag, edge_norm, 0, 255, cv2.NORM_MINMAX)
        thresh = cv2.threshold(edge_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morphed = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        
        return (x, y, w, h)

    def detect_text_orientation(self, img, label_region=None):
        """Detect text orientation focusing on the label region"""
        if label_region:
            x, y, w, h = label_region
            img = img[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        horizontal_score = 0
        vertical_score = 0
        text_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if 100 < area < 50000:
                aspect_ratio = w / h if h > 0 else 0
                weight = np.sqrt(area)
                
                if aspect_ratio > 2.5:
                    horizontal_score += weight
                    text_regions.append(('horizontal', w, h, area))
                elif aspect_ratio < 0.4:
                    vertical_score += weight
                    text_regions.append(('vertical', w, h, area))
        
        if len(text_regions) > 0:
            horizontal_count = sum(1 for r in text_regions if r[0] == 'horizontal')
            vertical_count = sum(1 for r in text_regions if r[0] == 'vertical')
            
            horizontal_score *= (1 + horizontal_count * 0.1)
            vertical_score *= (1 + vertical_count * 0.1)
        
        return horizontal_score, vertical_score

    def _edge_orientation_strength(self, img, label_region=None):
        """Return (vertical_edge_sum, horizontal_edge_sum) inside ROI.
        Vertical edges are measured with dx (Sobel x), horizontal with dy.
        """
        if label_region:
            x, y, w, h = label_region
            img = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vx = float(np.sum(np.abs(sx)))
        vy = float(np.sum(np.abs(sy)))
        return vx, vy

    def score_barcodes(self, img, label_region=None):
        """Score orientation by decoding barcodes in ROI if pyzbar is available.
        Returns (score, count). Higher score indicates more likely upright label.
        Prefers tall barcode bounding boxes (bars vertical when upright).
        """
        if label_region:
            x, y, w, h = label_region
            img = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not PYZBAR_AVAILABLE or zbar_decode is None:
            return 0.0, 0
        try:
            codes = zbar_decode(gray)
        except Exception:
            codes = []
        score = 0.0
        for c in codes:
            x, y, w, h = c.rect
            area = float(max(w, 1) * max(h, 1))
            aspect = (h + 1e-3) / (w + 1e-3)
            score += area * min(aspect, 4.0)
        return score, len(codes)

    def detect_barcode_orientation(self, img, label_region=None):
        """Return 'vertical', 'horizontal', or 'unknown' for barcodes in ROI.
        Uses pyzbar if available; falls back to edge orientation if none found.
        """
        if label_region:
            x, y, w, h = label_region
            img = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if PYZBAR_AVAILABLE and zbar_decode is not None:
            try:
                codes = zbar_decode(gray)
            except Exception:
                codes = []
            if codes:
                largest = max(codes, key=lambda c: c.rect.width * c.rect.height)
                w = largest.rect.width
                h = largest.rect.height
                if h > w * 1.1:
                    return 'vertical'
                elif w > h * 1.1:
                    return 'horizontal'
                else:
                    return 'unknown'
        vx, vy = self._edge_orientation_strength(img)
        if vx > vy * 1.1:
            return 'vertical'
        if vy > vx * 1.1:
            return 'horizontal'
        return 'unknown'

    def _adjust_rect_to_aspect(self, rect, img_shape, target_ratio):
        """Expand rect minimally so it has `target_ratio` (w/h) and stays in bounds.
        Keeps the original rect fully contained.
        rect: (x, y, w, h)
        img_shape: (H, W, C)
        """
        x, y, w, h = rect
        H, W = img_shape[0], img_shape[1]
        cur_ratio = w / max(h, 1)
        if cur_ratio < target_ratio:
            new_w = int(round(h * target_ratio))
            new_h = h
        else:
            new_w = w
            new_h = int(round(w / target_ratio))

        min_x = max(0, x + w - new_w)
        max_x = min(x, W - new_w)
        new_x = min(max(int(round(x + (w - new_w) / 2)), min_x), max_x)

        min_y = max(0, y + h - new_h)
        max_y = min(y, H - new_h)
        new_y = min(max(int(round(y + (h - new_h) / 2)), min_y), max_y)

        new_x = max(0, min(new_x, max(0, W - new_w)))
        new_y = max(0, min(new_y, max(0, H - new_h)))

        return new_x, new_y, new_w, new_h

    def find_largest_barcode_rect(self, img, label_region=None):
        """Return absolute union rect (x,y,w,h) for all decoded barcodes, or None.
        Backwards-compatible name; now returns the bounding box of all barcodes
        to better center multi-barcode labels.
        """
        if not PYZBAR_AVAILABLE or zbar_decode is None:
            return None
        x0 = y0 = 0
        roi = img
        if label_region:
            x0, y0, w0, h0 = label_region
            roi = img[y0:y0+h0, x0:x0+w0]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        try:
            codes = zbar_decode(gray)
        except Exception:
            codes = []
        if not codes:
            return None
        left = min(c.rect.left for c in codes)
        top = min(c.rect.top for c in codes)
        right = max(c.rect.left + c.rect.width for c in codes)
        bottom = max(c.rect.top + c.rect.height for c in codes)
        return (x0 + left, y0 + top, right - left, bottom - top)

    def second_pass_refine_rect(self, crop):
        """Refine crop on the cropped label image and return a tighter rect
        (rx, ry, rw, rh) relative to `crop`.
        Strategy:
        - If a border exists inside the crop, trim to inside the border (with a small inset).
        - Otherwise, binarize + morphology, union of sufficiently large components,
          and return that bounding box with a small safety margin.
        """
        H, W = crop.shape[:2]
        b = self.find_label_border(crop)
        if b is not None:
            bx, by, bw, bh = b
            inset = max(1, int(round(min(bw, bh) * 0.01)))
            rx = max(0, bx + inset)
            ry = max(0, by + inset)
            rw = max(1, min(W - rx, bw - 2 * inset))
            rh = max(1, min(H - ry, bh - 2 * inset))
            return rx, ry, rw, rh

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        k = max(5, int(round(min(H, W) * 0.015)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mor = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, kernel, iterations=1)
        cnts, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0, 0, W, H
        area_min = 0.003 * (H * W)
        xs, ys, xe, ye = W, H, 0, 0
        found = False
        for c in cnts:
            a = cv2.contourArea(c)
            if a < area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            xs = min(xs, x); ys = min(ys, y)
            xe = max(xe, x + w); ye = max(ye, y + h)
            found = True
        if not found:
            x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            xs, ys, xe, ye = x, y, x + w, y + h
        padx = max(1, int(round(0.02 * (xe - xs))))
        pady = max(1, int(round(0.02 * (ye - ys))))
        rx = max(0, xs - padx)
        ry = max(0, ys - pady)
        rw = min(W - rx, (xe - xs) + 2 * padx)
        rh = min(H - ry, (ye - ys) + 2 * pady)
        return rx, ry, rw, rh

    def _integral_sum(self, ii, x, y, w, h):
        x2 = x + w; y2 = y + h
        return ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x]

    def _refine_rect_density(self, img, bin_mask, orientation, rect):
        """Shift rect locally to maximize content density + oriented edges.
        Keeps size fixed; returns new (x,y,w,h).
        """
        x, y, w, h = rect
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if orientation == 'vertical':
            sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        else:
            sob = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.convertScaleAbs(np.abs(sob))
        ii_bin = cv2.integral(bin_mask.astype(np.uint8))
        ii_edge = cv2.integral(edge)

        max_dx = max(2, int(round(w * 0.08)))
        max_dy = max(2, int(round(h * 0.08)))
        step = max(2, int(round(min(w, h) * 0.02)))
        best = (x, y)
        best_score = -1.0
        norm = float(w * h)
        for dy in range(-max_dy, max_dy + 1, step):
            for dx in range(-max_dx, max_dx + 1, step):
                nx = min(max(0, x + dx), max(0, W - w))
                ny = min(max(0, y + dy), max(0, H - h))
                dark = self._integral_sum(ii_bin, nx, ny, w, h) / (255.0 * norm)
                edges = self._integral_sum(ii_edge, nx, ny, w, h) / (255.0 * norm)
                score = dark + 0.35 * edges
                if score > best_score:
                    best_score = score
                    best = (nx, ny)
        return best[0], best[1], w, h

    def _nudge_rect_by_strips(self, bin_mask, rect, img_shape):
        """Shift the rect if 10% strips outside have content being cut off.
        Keeps size fixed; returns new (x,y,w,h).
        Uses binary mask `bin_mask` (255=content) to measure densities.
        """
        x, y, w, h = rect
        H, W = img_shape[:2]
        ii = cv2.integral(bin_mask.astype(np.uint8))
        mx = max(1, int(round(w * 0.10)))
        my = max(1, int(round(h * 0.10)))

        def strip_density(sx, sy, sw, sh):
            if sw <= 0 or sh <= 0:
                return 0.0
            sx = max(0, min(sx, W))
            sy = max(0, min(sy, H))
            sw = max(0, min(sw, W - sx))
            sh = max(0, min(sh, H - sy))
            if sw == 0 or sh == 0:
                return 0.0
            s = self._integral_sum(ii, sx, sy, sw, sh)
            return float(s) / float(255 * sw * sh)

        left_den = strip_density(x - mx, y, mx, h)
        right_den = strip_density(x + w, y, mx, h)
        top_den = strip_density(x, y - my, w, my)
        bot_den = strip_density(x, y + h, w, my)

        eps = 0.01
        dx = 0
        dy = 0
        if left_den > right_den + eps:
            dx = -min(mx, int(round((left_den - right_den) * mx * 2)))
        elif right_den > left_den + eps:
            dx = min(mx, int(round((right_den - left_den) * mx * 2)))

        if top_den > bot_den + eps:
            dy = -min(my, int(round((top_den - bot_den) * my * 2)))
        elif bot_den > top_den + eps:
            dy = min(my, int(round((bot_den - top_den) * my * 2)))

        nx = min(max(0, x + dx), max(0, W - w))
        ny = min(max(0, y + dy), max(0, H - h))
        return nx, ny, w, h

    def _center_rect_on_content(self, bin_mask, rect, img_shape, expand_pct=0.10):
        """Recenter the rectangle so its center matches the content centroid
        within a slightly expanded analysis area around the current rect.
        Keeps size fixed; returns (x,y,w,h).
        """
        x, y, w, h = rect
        H, W = img_shape[:2]
        pad_x = max(1, int(round(w * expand_pct)))
        pad_y = max(1, int(round(h * expand_pct)))
        ax = max(0, x - pad_x)
        ay = max(0, y - pad_y)
        aw = min(W - ax, w + 2 * pad_x)
        ah = min(H - ay, h + 2 * pad_y)
        roi = bin_mask[ay:ay+ah, ax:ax+aw]
        if roi.size == 0:
            return x, y, w, h
        roi_bin = (roi > 0).astype(np.uint8)
        m = cv2.moments(roi_bin, binaryImage=True)
        if m["m00"] < 1.0:
            return x, y, w, h
        cx_local = m["m10"] / m["m00"]
        cy_local = m["m01"] / m["m00"]
        cx = ax + cx_local
        cy = ay + cy_local
        nx = int(round(cx - w / 2))
        ny = int(round(cy - h / 2))
        nx = min(max(0, nx), max(0, W - w))
        ny = min(max(0, ny), max(0, H - h))
        return nx, ny, w, h

    def _center_rect_horizontal(self, bin_mask, rect, img_shape, expand_pct=0.10):
        """Recenter horizontally only to the content centroid within a slightly
        expanded window. Keeps top/bottom as-is to preserve header exclusion.
        Returns (x,y,w,h).
        """
        x, y, w, h = rect
        H, W = img_shape[:2]
        pad_x = max(1, int(round(w * expand_pct)))
        pad_y = max(1, int(round(h * expand_pct)))
        ax = max(0, x - pad_x)
        ay = max(0, y - pad_y)
        aw = min(W - ax, w + 2 * pad_x)
        ah = min(H - ay, h + 2 * pad_y)
        roi = bin_mask[ay:ay+ah, ax:ax+aw]
        if roi.size == 0:
            return rect
        roi_bin = (roi > 0).astype(np.uint8)
        m = cv2.moments(roi_bin, binaryImage=True)
        if m["m00"] < 1.0:
            return rect
        cx_local = m["m10"] / m["m00"]
        cx = ax + cx_local
        nx = int(round(cx - w / 2))
        nx = min(max(0, nx), max(0, W - w))
        return nx, y, w, h

    def rotate_image(self, img, angle):
        """Rotate image by specified angle (90, 180, 270)"""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def pdf_to_image(self, pdf_bytes, dpi=300):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        get_pm = getattr(page, "get_pixmap", None)
        if get_pm is None:
            get_pm = getattr(page, "getPixmap", None)
        if get_pm is None:
            doc.close()
            raise AttributeError("PyMuPDF Page has no get_pixmap or getPixmap")
        pix = get_pm(matrix=mat, alpha=False)
        width = getattr(pix, "width", getattr(pix, "w", None))
        height = getattr(pix, "height", getattr(pix, "h", None))
        if width is None or height is None:
            doc.close()
            raise AttributeError("Pixmap missing width/height attributes")
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(height, width, 3)
        doc.close()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def process_pdf(self, pdf_bytes):
        """Process PDF using the EXACT original detection logic from detect_crop"""
        self.original_img = self.pdf_to_image(pdf_bytes, dpi=self.dpi)
        img = self.original_img.copy()
        
        x = y = w = h = 0
        used_border = False
        label_prefix = "Detected"
        
        border_rect = self.find_label_border(img)
        if border_rect is not None:
            bx, by, bw, bh = border_rect
            shrink = max(2, int(round(min(bw, bh) * 0.01)))
            x = max(0, bx + shrink)
            y = max(0, by + shrink)
            w = max(1, min(img.shape[1] - x, bw - 2 * shrink))
            h = max(1, min(img.shape[0] - y, bh - 2 * shrink))

            pad_x = max(1, int(round(w * 0.05)))
            pad_y = max(1, int(round(h * 0.05)))
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(img.shape[1] - x, w + 2 * pad_x)
            h = min(img.shape[0] - y, h + 2 * pad_y)
            used_border = True
            label_prefix = "Detected (border+10%)"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No label content detected. Try adjusting the threshold.")
        
        img_area = img.shape[0] * img.shape[1]
        valid_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            area_ratio = area / img_area
            if 0.20 < area_ratio < 0.95:
                valid_contours.append(c)
        
        if not valid_contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) / img_area < 0.95]
            if not valid_contours:
                raise ValueError("Could not identify label region. Try adjusting threshold.")
        
        if not used_border:
            contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

        if self.aspect_lock and not used_border:
            orientation = self.detect_barcode_orientation(img, (x, y, w, h))
            if orientation == 'vertical':
                target_ratio = 6.0 / 4.0
            elif orientation == 'horizontal':
                target_ratio = 4.0 / 6.0
            else:
                vx, vy = self._edge_orientation_strength(img, (x, y, w, h))
                target_ratio = 6.0 / 4.0 if vx >= vy else 4.0 / 6.0
            x, y, w, h = self._adjust_rect_to_aspect((x, y, w, h), img.shape, target_ratio)
            brect = self.find_largest_barcode_rect(img, (x, y, w, h))
            if brect is not None:
                bx, by, bw, bh = brect
                cx = bx + bw // 2
                cy = by + bh // 2
                x = min(max(0, cx - w // 2), max(0, img.shape[1] - w))
                y = min(max(0, cy - h // 2), max(0, img.shape[0] - h))
            orient = 'vertical' if target_ratio >= 1.0 else 'horizontal'
            x, y, w, h = self._refine_rect_density(img, thresh, orient, (x, y, w, h))
            x, y, w, h = self._nudge_rect_by_strips(thresh, (x, y, w, h), img.shape)
            x, y, w, h = self._center_rect_on_content(thresh, (x, y, w, h), img.shape, expand_pct=0.10)
            cut_y = self.detect_cut_line_y(img)
            if cut_y is not None:
                margin = max(5, int(round(0.01 * img.shape[0])))
                min_top = cut_y + margin
                if y < min_top:
                    y = min(img.shape[0] - h, min_top)
            x, y, w, h = self._align_rect_to_vertical_edges(img, (x, y, w, h))
            x, y, w, h = self._anchor_rect_to_barcodes(img, (x, y, w, h), margin_frac=0.02)
            brect_final = self.find_largest_barcode_rect(img, None)
            cut_y2 = self.detect_cut_line_y(img)
            min_top = 0 if cut_y2 is None else min(img.shape[0] - h, cut_y2 + max(5, int(0.01 * img.shape[0])))
            if brect_final is not None:
                bx, by, bw, bh = brect_final
                header_allow = max(6, int(round(0.10 * h)))
                cap_top = max(min_top, by - header_allow)
                y = max(y, cap_top)
            else:
                up = max(1, int(round(0.02 * h)))
                y = max(min_top, y - up)
            x, y, w, h = self._center_rect_horizontal(thresh, (x, y, w, h), img.shape, expand_pct=0.10)

            add_px = max(1, int(round(w * 0.05)))
            add_py = max(1, int(round(h * 0.05)))
            x = max(0, x - add_px)
            y = max(0, y - add_py)
            w = min(img.shape[1] - x, w + 2 * add_px)
            h = min(img.shape[0] - y, h + 2 * add_py)

        if not used_border:
            sub = img[y:y+h, x:x+w]
            rx, ry, rw, rh = self.second_pass_refine_rect(sub)
            x, y, w, h = x + rx, y + ry, rw, rh
        
        pad_percent = self.padding_percent / 100.0
        pad = int(min(w, h) * pad_percent)
        x, y = max(x - pad, 0), max(y - pad, 0)
        w = min(img.shape[1] - x, w + 2 * pad)
        h = min(img.shape[0] - y, h + 2 * pad)
        
        self.cropped_img = img[y:y + h, x:x + w]
        
        preview = img.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), max(5, int(min(w, h) * 0.005)))
        
        text = f"{label_prefix}: {w}x{h}px"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w, h) / 1000)
        thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = max(y - 10, text_size[1] + 10)
        
        cv2.rectangle(preview, (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)
        cv2.putText(preview, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        return self.cropped_img, preview, (x, y, w, h)
