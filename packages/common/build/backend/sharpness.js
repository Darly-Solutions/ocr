import cv from '@techstark/opencv-js';
export function measureSharpness(image) {
    const mat = cv.matFromImageData(image);
    const gray = new cv.Mat();
    cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    const laplacian = new cv.Mat();
    cv.Laplacian(gray, laplacian, cv.CV_64F);
    const mean = new cv.Mat();
    const stddev = new cv.Mat();
    cv.meanStdDev(laplacian, mean, stddev);
    const sharpness = stddev.data64F[0] ** 2; // variance = stddev^2
    mat.delete();
    gray.delete();
    laplacian.delete();
    mean.delete();
    stddev.delete();
    return sharpness;
}
//# sourceMappingURL=sharpness.js.map