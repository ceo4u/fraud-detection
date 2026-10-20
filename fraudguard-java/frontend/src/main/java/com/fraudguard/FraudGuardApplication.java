package com.fraudguard;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class FraudGuardApplication {
    public static void main(String[] args) {
        SpringApplication.run(FraudGuardApplication.class, args);
    }
}
