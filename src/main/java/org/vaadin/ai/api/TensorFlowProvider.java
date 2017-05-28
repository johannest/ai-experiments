package org.vaadin.ai.api;

import org.apache.commons.lang3.tuple.*;
import org.tensorflow.*;

import java.io.*;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;

/**
 * Copied and modified from Apache 2.0 licensed code of TensorFlow examples:
 * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 */
public class TensorFlowProvider {

    private List<String> labels;
    private byte[] graphDef;

    public TensorFlowProvider(String modelPath, String labelsPath) {
        loadModelAndLabels(modelPath, labelsPath);
    }

    private void loadModelAndLabels(String modelPath, String labelsPath) {
        graphDef = readAllBytesOrExit(Paths.get(modelPath));
        labels = readAllLinesOrExit(Paths.get(labelsPath));
    }

    public Pair<String, Float> classify(byte[] imageBytes) throws UnsupportedEncodingException {
        try (Tensor image = constructAndExecuteGraphToNormalizeImage(
                imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int bestLabelIdx = maxIndex(labelProbabilities);
            return Pair.of(labels.get(bestLabelIdx),labelProbabilities[bestLabelIdx] * 100f);
        }
    }

    private static Tensor constructAndExecuteGraphToNormalizeImage(
            byte[] imageBytes) {
        try (Graph graph = new Graph()) {
            GraphBuilder graphBuilder = new GraphBuilder(graph);
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were
            // converted to
            // float using (value - Mean)/Scale.
            final int height = 224;
            final int width = 224;
            final float mean = 117f;
            final float scale = 1f;

            // Since the graph is being constructed once per execution here, we
            // can use a constant for the
            // input image. If the graph were to be re-used for multiple input
            // images, a placeholder would
            // have been more appropriate.
            final Output input = graphBuilder.constant("input", imageBytes);
            final Output output = graphBuilder
                    .div(graphBuilder.sub(
                            graphBuilder.resizeBilinear(
                                    graphBuilder.expandDims(
                                            graphBuilder.cast(graphBuilder.decodeJpeg(input, 3),
                                                    DataType.FLOAT),
                                            graphBuilder.constant("make_batch", 0)),
                                    graphBuilder.constant("size", new int[] { height, width })),
                            graphBuilder.constant("mean", mean)),
                            graphBuilder.constant("scale", scale));
            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0);
            }
        }
    }

    private static float[] executeInceptionGraph(byte[] graphDef,
            Tensor imageTensor) {
        try (Graph graph = new Graph()) {
            graph.importGraphDef(graphDef);
            try (Session session = new Session(graph);
                    Tensor result = session.runner().feed("input", imageTensor)
                            .fetch("output").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(String.format(
                            "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                            Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println(
                    "Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println(
                    "Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    // In the fullness of time, equivalents of the methods of this class should
    // be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is
    // done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {
        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }

        Output resizeBilinear(Output images, Output size) {
            return binaryOp("ResizeBilinear", images, size);
        }

        Output expandDims(Output input, Output dim) {
            return binaryOp("ExpandDims", input, dim);
        }

        Output cast(Output value, DataType dtype) {
            return g.opBuilder("Cast", "Cast").addInput(value)
                    .setAttr("DstT", dtype).build().output(0);
        }

        Output decodeJpeg(Output contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents)
                    .setAttr("channels", channels).build().output(0);
        }

        Output decodePng(Output contents, long channels) {
            return g.opBuilder("DecodePng", "DecodePng").addInput(contents)
                    .setAttr("channels", channels).build().output(0);
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name).setAttr("dtype", t.dataType())
                        .setAttr("value", t).build().output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build()
                    .output(0);
        }

        private Graph g;
    }
}
