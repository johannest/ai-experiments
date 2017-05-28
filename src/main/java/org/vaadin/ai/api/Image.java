package org.vaadin.ai.api;


public class Image {

    private String imageData;
    private String mimeType;

    public String getImageData() {
        return imageData;
    }

    public String getMimeType() {
        return mimeType;
    }

    public void setImageData(String imageData) {
        this.imageData = imageData;
    }

    public void setMimeType(String mimeType) {
        this.mimeType = mimeType;
    }
}
