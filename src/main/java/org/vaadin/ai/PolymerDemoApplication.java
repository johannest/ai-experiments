package org.vaadin.ai;

import org.springframework.beans.factory.*;
import org.springframework.boot.*;
import org.springframework.boot.autoconfigure.*;
import org.springframework.context.annotation.*;

@SpringBootApplication
@Configuration
@ComponentScan
public class PolymerDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(PolymerDemoApplication.class, args);
    }

    @Bean
    public InitializingBean initializeApp() {
        return new InitializingBean() {

            @Override
            public void afterPropertiesSet() {

            }
        };
    }
}
