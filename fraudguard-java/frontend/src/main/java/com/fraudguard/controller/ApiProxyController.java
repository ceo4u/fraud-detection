package com.fraudguard.controller;

import com.fraudguard.model.*;
import com.fraudguard.service.FraudApiClient;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class ApiProxyController {

    private final FraudApiClient apiClient;

    public ApiProxyController(FraudApiClient apiClient) {
        this.apiClient = apiClient;
    }

    @GetMapping("/providers")
    public List<ProviderPrediction> getProviders(
            @RequestParam(defaultValue = "50") int limit,
            @RequestParam(defaultValue = "0.0", name = "min_prob") double minProb,
            @RequestParam(defaultValue = "desc") String sort
    ) {
        return apiClient.getProviders(limit, minProb, sort);
    }

    @GetMapping("/providers/count")
    public Map<String, Object> getProviderCount() {
        return apiClient.getProviderCount();
    }

    @PostMapping("/predict")
    public PredictionResponse predict(@RequestBody Map<String, Object> payload) {
        return apiClient.predict(payload);
    }

    @GetMapping("/explain/{providerId}")
    public ShapExplanation explain(@PathVariable String providerId) {
        return apiClient.explain(providerId);
    }

    @GetMapping("/metrics")
    public ModelMetrics getMetrics() {
        return apiClient.getMetrics();
    }

    @GetMapping("/demo-provider")
    public Map<String, Object> getDemoProvider() {
        return apiClient.getDemoProvider();
    }

    @GetMapping("/plots/{name}")
    public ResponseEntity<byte[]> getPlot(@PathVariable String name) {
        byte[] imageBytes = apiClient.getPlot(name);
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_TYPE, MediaType.IMAGE_PNG_VALUE)
                .body(imageBytes);
    }
}
