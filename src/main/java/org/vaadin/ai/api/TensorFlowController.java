package org.vaadin.ai.api;

import org.apache.commons.lang3.tuple.*;
import org.springframework.http.*;
import org.springframework.stereotype.*;
import org.springframework.util.*;
import org.springframework.web.bind.annotation.*;

import javax.imageio.*;
import javax.xml.bind.*;
import java.awt.image.*;
import java.io.*;

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
        String path = TensorFlowController.class.getResource("../../../../tf/").getPath();
        System.out.println(">>>>"+path);
        return path;
    }

    @RequestMapping(method = RequestMethod.POST)
    public ResponseEntity<Void> predictClass(@RequestBody Image image) {
        try {
            byte[] jpgData = getJpgBytes(getPngBytes(image.getImageData()));

            Pair<String, Float> res = tf.classify(jpgData);

            System.out.println(res.getKey()+" "+res.getRight());

            MultiValueMap<String, String> headers = new LinkedMultiValueMap();
            headers.add("result",res.getKey());
            headers.add("prob",Double.toString(res.getRight()));
            final ResponseEntity<Void> response = new ResponseEntity<>(headers, HttpStatus.ACCEPTED);
            return response;
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<Void>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    public static byte[] getPngBytes(String imgData) throws UnsupportedEncodingException {
        return DatatypeConverter.parseBase64Binary(imgData.split(",")[1]);
    }

    public static byte[] getJpgBytes(byte[] pngBytes) throws IOException {
        final BufferedImage bimg = ImageIO.read(
                new ByteArrayInputStream(pngBytes));
        final ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(bimg, "jpg", output);
        byte[] jpgData = output.toByteArray();
        return jpgData;
    }
}
