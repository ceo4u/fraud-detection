package com.fraudguard.service;

import com.fraudguard.config.AppProperties;
import com.fraudguard.exception.BackendUnavailableException;
import com.fraudguard.model.*;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class FraudApiClient {

    private final RestTemplate restTemplate;
    private final AppProperties appProperties;

    public FraudApiClient(RestTemplate restTemplate, AppProperties appProperties) {
        this.restTemplate = restTemplate;
        this.appProperties = appProperties;
    }

    private String getBaseUrl() {
        return appProperties.getBackend().getUrl();
    }

    public List<ProviderPrediction> getProviders(int limit, double minProb, String sort) {
        String url = String.format("%s/providers?limit=%d&min_prob=%.4f&sort=%s", getBaseUrl(), limit, minProb, sort);
        try {
            ResponseEntity<List<ProviderPrediction>> response = restTemplate.exchange(
                    url,
                    HttpMethod.GET,
                    null,
                    new ParameterizedTypeReference<List<ProviderPrediction>>() {}
            );
            return response.getBody() != null ? response.getBody() : Collections.emptyList();
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to fetch provider list from backend: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getProviderCount() {
        String url = getBaseUrl() + "/providers/count";
        try {
            return restTemplate.getForObject(url, Map.class);
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to fetch provider counts from backend: " + ex.getMessage(), ex);
        }
    }

    public PredictionResponse predict(Map<String, Object> features) {
        String url = getBaseUrl() + "/predict";
        try {
            PredictionRequest request = new PredictionRequest(features);
            return restTemplate.postForObject(url, request, PredictionResponse.class);
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to get prediction from backend: " + ex.getMessage(), ex);
        }
    }

    public ShapExplanation explain(String providerId) {
        String url = getBaseUrl() + "/explain/" + providerId;
        try {
            return restTemplate.getForObject(url, ShapExplanation.class);
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to fetch SHAP explanation for " + providerId + ": " + ex.getMessage(), ex);
        }
    }

    public ModelMetrics getMetrics() {
        String url = getBaseUrl() + "/metrics";
        try {
            return restTemplate.getForObject(url, ModelMetrics.class);
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to fetch model metrics from backend: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getDemoProvider() {
        String url = getBaseUrl() + "/demo-provider";
        try {
            return restTemplate.getForObject(url, Map.class);
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to fetch demo provider from backend: " + ex.getMessage(), ex);
        }
    }

    public byte[] getPlot(String name) {
        String url = getBaseUrl() + "/plots/" + name;
        try {
            return restTemplate.getForObject(url, byte[].class);
        } catch (RestClientException ex) {
            throw new BackendUnavailableException("Failed to fetch report plot " + name + ": " + ex.getMessage(), ex);
        }
    }
}
