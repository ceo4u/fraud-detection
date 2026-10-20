package com.fraudguard.exception;

import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BackendUnavailableException.class)
    public String handleBackendUnavailable(BackendUnavailableException ex, Model model) {
        model.addAttribute("backendAvailable", false);
        model.addAttribute("errorMessage", ex.getMessage());
        return "error";
    }

    @ExceptionHandler(Exception.class)
    public String handleGenericException(Exception ex, Model model) {
        model.addAttribute("backendAvailable", false);
        model.addAttribute("errorMessage", "Unexpected error: " + ex.getMessage());
        return "error";
    }
}
