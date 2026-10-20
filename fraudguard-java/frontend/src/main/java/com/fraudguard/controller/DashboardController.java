package com.fraudguard.controller;

import com.fraudguard.exception.BackendUnavailableException;
import com.fraudguard.model.ModelMetrics;
import com.fraudguard.model.ProviderPrediction;
import com.fraudguard.service.FraudApiClient;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.*;

@Controller
public class DashboardController {

    private final FraudApiClient apiClient;

    public DashboardController(FraudApiClient apiClient) {
        this.apiClient = apiClient;
    }

    private boolean populateCommonAttributes(Model model, String activeTab) {
        model.addAttribute("activeTab", activeTab);
        try {
            ModelMetrics metrics = apiClient.getMetrics();
            Map<String, Object> counts = apiClient.getProviderCount();
            model.addAttribute("metrics", metrics);
            model.addAttribute("providerCount", counts);
            model.addAttribute("backendAvailable", true);
            return true;
        } catch (BackendUnavailableException ex) {
            model.addAttribute("backendAvailable", false);
            model.addAttribute("errorMessage", ex.getMessage());
            return false;
        }
    }

    @GetMapping("/")
    public String dashboard(Model model) {
        boolean ok = populateCommonAttributes(model, "dashboard");
        if (!ok) {
            return "error";
        }
        try {
            List<ProviderPrediction> providers = apiClient.getProviders(50, 0.0, "desc");
            model.addAttribute("providers", providers);
            return "dashboard";
        } catch (BackendUnavailableException ex) {
            model.addAttribute("backendAvailable", false);
            model.addAttribute("errorMessage", ex.getMessage());
            return "error";
        }
    }

    @GetMapping("/explain")
    public String explain(
            @RequestParam(name = "provider", required = false) String provider,
            Model model
    ) {
        boolean ok = populateCommonAttributes(model, "explain");
        if (!ok) {
            return "error";
        }
        try {
            Map<String, Object> demo = apiClient.getDemoProvider();
            String defaultProvider = (provider != null && !provider.isBlank())
                    ? provider
                    : (String) demo.getOrDefault("provider", "PRV52340");
            model.addAttribute("selectedProvider", defaultProvider);
            model.addAttribute("demoProvider", demo);

            // Fetch top 25 providers for dropdown selector
            List<ProviderPrediction> topProviders = apiClient.getProviders(25, 0.0, "desc");
            model.addAttribute("topProviders", topProviders);

            return "explain";
        } catch (BackendUnavailableException ex) {
            model.addAttribute("backendAvailable", false);
            model.addAttribute("errorMessage", ex.getMessage());
            return "error";
        }
    }

    @GetMapping("/metrics")
    public String metrics(Model model) {
        boolean ok = populateCommonAttributes(model, "metrics");
        if (!ok) {
            return "error";
        }
        return "metrics";
    }

    @GetMapping("/about")
    public String about(Model model) {
        boolean ok = populateCommonAttributes(model, "about");
        if (!ok) {
            return "error";
        }
        return "about";
    }
}
