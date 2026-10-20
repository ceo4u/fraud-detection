package com.fraudguard.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

public class ShapExplanation {
    @JsonProperty("provider")
    private String provider;

    @JsonProperty("base_value")
    private Double baseValue;

    @JsonProperty("shap_values")
    private Map<String, Double> shapValues;

    @JsonProperty("feature_values")
    private Map<String, Double> featureValues;

    @JsonProperty("top_drivers")
    private List<Driver> topDrivers;

    @JsonProperty("model_probability")
    private Double modelProbability;

    public static class Driver {
        private String feature;
        private Double value;
        private Double shap;

        public Driver() {}

        public String getFeature() {
            return feature;
        }

        public void setFeature(String feature) {
            this.feature = feature;
        }

        public Double getValue() {
            return value;
        }

        public void setValue(Double value) {
            this.value = value;
        }

        public Double getShap() {
            return shap;
        }

        public void setShap(Double shap) {
            this.shap = shap;
        }
    }

    public ShapExplanation() {}

    public String getProvider() {
        return provider;
    }

    public void setProvider(String provider) {
        this.provider = provider;
    }

    public Double getBaseValue() {
        return baseValue;
    }

    public void setBaseValue(Double baseValue) {
        this.baseValue = baseValue;
    }

    public Map<String, Double> getShapValues() {
        return shapValues;
    }

    public void setShapValues(Map<String, Double> shapValues) {
        this.shapValues = shapValues;
    }

    public Map<String, Double> getFeatureValues() {
        return featureValues;
    }

    public void setFeatureValues(Map<String, Double> featureValues) {
        this.featureValues = featureValues;
    }

    public List<Driver> getTopDrivers() {
        return topDrivers;
    }

    public void setTopDrivers(List<Driver> topDrivers) {
        this.topDrivers = topDrivers;
    }

    public Double getModelProbability() {
        return modelProbability;
    }

    public void setModelProbability(Double modelProbability) {
        this.modelProbability = modelProbability;
    }
}
