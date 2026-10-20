package com.fraudguard.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public class PredictionRequest {
    @JsonProperty("features")
    private Map<String, Object> features;

    public PredictionRequest() {}

    public PredictionRequest(Map<String, Object> features) {
        this.features = features;
    }

    public Map<String, Object> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Object> features) {
        this.features = features;
    }
}
