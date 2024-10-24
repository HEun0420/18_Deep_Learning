package com.ohgiraffers.apirequest.section01;
import com.ohgiraffers.apirequest.section01.dto.RequestDTO;
import com.ohgiraffers.apirequest.section01.dto.ResponseDTO;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
/*
 * WebClient
 *
 * Sprint WebFlux에 포함되어있는 HTTP 요청 라이브러리
 * Reactor(반응형) 기반의 API를 가지고 있어,
 * 기본적으로 비동기 방식이지만 동기방식도 지원한다.
 * */
@Service
@Slf4j
public class WebClientService {
    private final WebClient webClient;
    private final String FAST_API_SERVER_URL = "http://localhost:8000";
    public WebClientService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
                .baseUrl(FAST_API_SERVER_URL)
                .build();
    }
    public ResponseDTO translateText(RequestDTO requestDTO) {
        ResponseDTO responseDTO = webClient.post()
                .uri("/translate")
                .bodyValue(requestDTO)
                .retrieve() // 요청 보내기
                .bodyToMono(ResponseDTO.class) // 응답 받을 값을 변환
                .doOnSuccess(response -> log.info("번역완료"))
                .doOnError(error -> log.error("번역 API 호출중 에러발생"))
                .block(); // 비동기 작업을 동기식으로 변환
        return responseDTO;
    }
}