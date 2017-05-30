package org.vaadin.ai.api;

import org.apache.commons.lang3.tuple.Pair;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import java.util.Base64;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

@Service
@RestController
@RequestMapping("/api/tensorflow")
public class TensorFlowController {

    final TensorFlowProvider tf;

    public TensorFlowController() {
        tf = new TensorFlowProvider(
                getTFDirectoryPath() + "tensorflow_inception_graph.pb",
                getTFDirectoryPath() + "imagenet_comp_graph_label_strings.txt");
    }

    private String getTFDirectoryPath() {
        return TensorFlowController.class.getResource("../../../../tf/")
                .getPath();
    }

    @RequestMapping(method = RequestMethod.POST)
    public ResponseEntity<Void> predictClass(@RequestBody Image image) {
        try {
            byte[] bytes = getBytesFromBase64(image.getImageData());
            if (image.getMimeType().contains("png")) {
                bytes = convertPngBytesToJpgBytes(bytes);
            }
            Pair<String, Float> res = tf.classify(bytes);

            MultiValueMap<String, String> headers = new LinkedMultiValueMap();
            headers.add("result", res.getKey());
            headers.add("prob", Float.toString(res.getRight()));
            final ResponseEntity<Void> response = new ResponseEntity<>(headers,
                    HttpStatus.ACCEPTED);
            return response;
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<Void>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    public static byte[] getBytesFromBase64(String imgData)
            throws UnsupportedEncodingException {
        final String base64Data = imgData.split(",")[1];
        return Base64.getDecoder().decode(base64Data);
    }

    public static byte[] convertPngBytesToJpgBytes(byte[] pngBytes)
            throws IOException {
        final BufferedImage bimg = ImageIO
                .read(new ByteArrayInputStream(pngBytes));
        final ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(bimg, "jpg", output);
        byte[] jpgData = output.toByteArray();
        return jpgData;
    }
}