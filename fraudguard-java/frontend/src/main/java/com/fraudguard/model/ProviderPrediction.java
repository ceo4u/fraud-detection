package com.fraudguard.model;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonProperty;

public class ProviderPrediction {
    @JsonProperty("provider")
    @JsonAlias({"Provider", "provider"})
    private String provider;

    @JsonProperty("y_true")
    private Integer yTrue;

    @JsonProperty("y_proba")
    private Double yProba;

    @JsonProperty("y_pred")
    private Integer yPred;

    @JsonProperty("risk_tier")
    private String riskTier;

    public ProviderPrediction() {}

    public String getProvider() {
        return provider;
    }

    public void setProvider(String provider) {
        this.provider = provider;
    }

    public Integer getyTrue() {
        return yTrue;
    }

    public void setyTrue(Integer yTrue) {
        this.yTrue = yTrue;
    }

    public Double getyProba() {
        return yProba;
    }

    public void setyProba(Double yProba) {
        this.yProba = yProba;
    }

    public Integer getyPred() {
        return yPred;
    }

    public void setyPred(Integer yPred) {
        this.yPred = yPred;
    }

    public String getRiskTier() {
        return riskTier;
    }

    public void setRiskTier(String riskTier) {
        this.riskTier = riskTier;
    }

    public String getProbaDecimal() {
        if (yProba == null) return "0.0000";
        return String.format(java.util.Locale.US, "%.4f", yProba);
    }

    public String getProbaFormatted() {
        if (yProba == null) return "0.00%";
        return String.format("%.2f%%", yProba * 100);
    }

    public int getProbaPercent() {
        if (yProba == null) return 0;
        return (int) Math.round(yProba * 100);
    }

    public String getTierClass() {
        if (riskTier == null) return "low";
        switch (riskTier.toLowerCase()) {
            case "high": return "high";
            case "medium": return "med";
            default: return "low";
        }
    }
}
