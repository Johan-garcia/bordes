#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// Detecta sombras usando el canal V (Value) del espacio HSV:
// Las sombras tienen bajo brillo (V) pero saturación similar al fondo
// ─────────────────────────────────────────────────────────────────────────────
Mat detectarSombras(const Mat& img_bgr, int umbral_v, int umbral_s) {
    Mat hsv;
    cvtColor(img_bgr, hsv, COLOR_BGR2HSV);

    vector<Mat> canales;
    split(hsv, canales);
    Mat& V = canales[2];  // canal de brillo
    Mat& S = canales[1];  // canal de saturación

    // Sombra: brillo bajo Y saturación moderada (no es fondo blanco/negro puro)
    Mat mask_v, mask_s, sombra;
    threshold(V, mask_v, umbral_v, 255, THRESH_BINARY_INV);  // bajo brillo
    threshold(S, mask_s, umbral_s, 255, THRESH_BINARY);       // algo de color
    bitwise_and(mask_v, mask_s, sombra);

    // Limpiar con morfología
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(sombra, sombra, MORPH_OPEN,  kernel);
    morphologyEx(sombra, sombra, MORPH_CLOSE, kernel);

    return sombra;
}

// ─────────────────────────────────────────────────────────────────────────────
// Aplica filtro Sobel y retorna magnitud del gradiente normalizada
// ─────────────────────────────────────────────────────────────────────────────
Mat aplicarSobel(const Mat& gray) {
    Mat grad_x, grad_y, abs_x, abs_y, grad;

    Sobel(gray, grad_x, CV_16S, 1, 0, 3);  // gradiente horizontal
    Sobel(gray, grad_y, CV_16S, 0, 1, 3);  // gradiente vertical

    convertScaleAbs(grad_x, abs_x);
    convertScaleAbs(grad_y, abs_y);

    addWeighted(abs_x, 0.5, abs_y, 0.5, 0, grad);  // magnitud aproximada
    return grad;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // ── Cargar imagen ─────────────────────────────────────────────────────────
    string ruta = (argc > 1) ? argv[1] : "../data/imagen.png";
    Mat img = imread(ruta, IMREAD_COLOR);

    if (img.empty()) {
        cerr << "Error: no se encontró " << ruta << "\n"
             << "Uso: ./taller_sombras ../data/imagen.png\n";
        return -1;
    }

    cout << "Imagen cargada: " << img.cols << "x" << img.rows << "\n";
    cout << "Teclas: 'q' salir | 's' guardar resultados\n";

    // ── Convertir a gris para Sobel ───────────────────────────────────────────
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // ── Parámetros ────────────────────────────────────────────────────────────
    int umbral_v    = 80;   // umbral de brillo para detectar sombra
    int umbral_s    = 30;   // umbral de saturación mínima
    int umbral_bin  = 100;  // umbral de binarización global

    // ── Pipeline completo ─────────────────────────────────────────────────────

    // 1. Binarización global con umbral
    Mat binaria;
    threshold(gray, binaria, umbral_bin, 255, THRESH_BINARY);

    // 2. Detección de sombras (espacio HSV)
    Mat mascara_sombra = detectarSombras(img, umbral_v, umbral_s);

    // 3. Eliminar sombras de la máscara binaria
    Mat sin_sombra;
    bitwise_and(binaria, ~mascara_sombra, sin_sombra);

    // 4. Sobel sobre imagen original
    Mat sobel_original = aplicarSobel(gray);

    // 5. Sobel sobre imagen sin sombras
    //    Primero enmascarar la imagen gris
    Mat gray_sin_sombra;
    gray.copyTo(gray_sin_sombra, ~mascara_sombra);
    Mat sobel_sin_sombra = aplicarSobel(gray_sin_sombra);

    // 6. Umbral sobre Sobel (bordes fuertes solamente)
    Mat sobel_umbral;
    threshold(sobel_original, sobel_umbral, 50, 255, THRESH_BINARY);

    // ── Visualización ─────────────────────────────────────────────────────────
    // Overlay de sombras en azul sobre imagen original
    Mat overlay = img.clone();
    overlay.setTo(Scalar(200, 50, 50), mascara_sombra);
    addWeighted(img, 0.6, overlay, 0.4, 0, overlay);

    // Convertir máscaras a BGR para mostrar en color
    Mat sombra_bgr, binaria_bgr, sin_sombra_bgr, sobel_umbral_bgr;
    cvtColor(mascara_sombra,  sombra_bgr,      COLOR_GRAY2BGR);
    cvtColor(binaria,         binaria_bgr,     COLOR_GRAY2BGR);
    cvtColor(sin_sombra,      sin_sombra_bgr,  COLOR_GRAY2BGR);
    cvtColor(sobel_umbral,    sobel_umbral_bgr,COLOR_GRAY2BGR);

    // Mostrar ventanas
    imshow("1 - Imagen Original",             img);
    imshow("2 - Binarizacion (umbral global)", binaria_bgr);
    imshow("3 - Mascara de Sombras (HSV)",     sombra_bgr);
    imshow("4 - Binarizacion sin Sombras",     sin_sombra_bgr);
    imshow("5 - Sobel Original",               sobel_original);
    imshow("6 - Sobel sin Sombras",            sobel_sin_sombra);
    imshow("7 - Sobel Umbralizado",            sobel_umbral_bgr);
    imshow("8 - Overlay Sombras",              overlay);

    // Estadísticas en consola
    cout << "\n=== ESTADÍSTICAS ===\n";
    cout << "Píxeles totales:         " << img.rows * img.cols << "\n";
    cout << "Píxeles detectados sombra: " << countNonZero(mascara_sombra) << "\n";
    cout << "% sombra:                "
         << 100.0 * countNonZero(mascara_sombra) / (img.rows * img.cols)
         << " %\n";
    cout << "Bordes Sobel detectados: " << countNonZero(sobel_umbral) << "\n";

    int key = waitKey(0);

    // Guardar resultados con tecla 's'
    if (key == 's') {
        imwrite("../data/resultado_binaria.png",      binaria);
        imwrite("../data/resultado_sombra.png",       mascara_sombra);
        imwrite("../data/resultado_sin_sombra.png",   sin_sombra);
        imwrite("../data/resultado_sobel.png",         sobel_original);
        imwrite("../data/resultado_sobel_umbral.png",  sobel_umbral);
        imwrite("../data/resultado_overlay.png",       overlay);
        cout << "Resultados guardados en data/\n";
    }

    destroyAllWindows();
    return 0;
}
