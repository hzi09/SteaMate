from django.shortcuts import render, redirect
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import User
from .serializers import (CreateUserSerializer, UserUpdateSerializer,
                          SteamSignupSerializer)
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework import permissions
from django.shortcuts import get_object_or_404
from django.conf import settings
import urllib.parse
from rest_framework_simplejwt.tokens import RefreshToken
import requests
from rest_framework_simplejwt.exceptions import InvalidToken
from rest_framework_simplejwt.exceptions import TokenError


class SignupAPIView(APIView):
    """일반 사용자 회원가입 API"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = CreateUserSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


class SteamLoginAPIView(APIView):
    """Steam OpenID 로그인 요청"""
    permission_classes = [AllowAny]

    def get(self, request):
        """GET 요청 시 Steam 로그인 페이지로 리디렉션"""
        steam_openid_url = "https://steamcommunity.com/openid/login"
        
        params = {
            "openid.ns": "http://specs.openid.net/auth/2.0",
            "openid.mode": "checkid_setup",
            "openid.return_to": f"{settings.SITE_URL}/api/v1/account/steam-callback/",
            "openid.realm": settings.SITE_URL,
            "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
            "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
        }

        steam_login_url = f"{steam_openid_url}?{urllib.parse.urlencode(params)}"
        return redirect(steam_login_url)
            
class SteamCallbackAPIView(APIView):
    """Steam 로그인 Callback API (Steam ID 검증)"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Steam 로그인 성공 후, OpenID 검증"""

        # GET 파라미터를 dict 형태로 변환
        openid_params = request.GET.dict()
        
        # 필수 OpenID 파라미터 유지
        steam_openid_params = {
            "openid.ns": openid_params.get("openid.ns", ""),
            "openid.mode": "check_authentication",
            "openid.op_endpoint": openid_params.get("openid.op_endpoint", ""),
            "openid.claimed_id": openid_params.get("openid.claimed_id", ""),
            "openid.identity": openid_params.get("openid.identity", ""),
            "openid.return_to": openid_params.get("openid.return_to", ""),
            "openid.response_nonce": openid_params.get("openid.response_nonce", ""),
            "openid.assoc_handle": openid_params.get("openid.assoc_handle", ""),
            "openid.signed": openid_params.get("openid.signed", ""),
            "openid.sig": openid_params.get("openid.sig", ""),
        }

        steam_openid_url = "https://steamcommunity.com/openid/login"

        # Steam OpenID 검증 요청 (POST 사용)
        response = requests.post(steam_openid_url, data=steam_openid_params)

        # Steam 응답 처리
        response_text = response.text.strip()
        print("Steam OpenID 응답 (첫 50자):", response_text[:50])

        # Steam 인증 실패 시
        if "is_valid:true" not in response_text:
            return Response(
                {"error": "Steam 인증 실패", "steam_response": response_text[:200]},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Steam ID 추출 (예외 처리 강화)
        steam_id_url = openid_params.get("openid.claimed_id", "")
        if not steam_id_url or not steam_id_url.startswith("https://steamcommunity.com/openid/id/"):
            return Response({"error": "잘못된 Steam ID 응답"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            steam_id = steam_id_url.split("/")[-1]
            if not steam_id.isdigit():
                raise ValueError("Steam ID가 숫자가 아닙니다.")
        except Exception as e:
            return Response({"error": f"Steam ID 처리 중 오류 발생: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # DB에서 해당 Steam ID가 존재하는지 확인
        user = User.objects.filter(steam_id=steam_id).first()

        if user:
            # 기존 회원이면 JWT 발급 후 로그인 처리
            refresh = RefreshToken.for_user(user)
            return Response({
                "message": "Steam 로그인 성공",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user_id": user.id,
                "redirect_url": "/"  # 홈으로 리다이렉트
            }, status=status.HTTP_200_OK)
        
        # 신규 회원이면 추가 정보 입력 필요 → 회원가입 페이지로 리다이렉트
        return Response({
            "message": "Steam 인증 성공. 추가 정보 입력이 필요합니다.",
            "steam_id": steam_id,
            "needs_update": True,
            "redirect_url": "/signup"  # 회원가입 페이지로 이동
        }, status=status.HTTP_201_CREATED)



class SteamSignupAPIView(APIView):
    """Steam 회원가입 (추가 정보 입력)"""
    permission_classes = [AllowAny]

    def post(self, request):
        """Steam 회원가입: 추가 정보 입력 후 계정 생성"""
        serializer = SteamSignupSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.save()

            # JWT 토큰 발급
            refresh = RefreshToken.for_user(user)
            response_data = serializer.data
            return Response({
                **serializer.data,  # 기존 serializer 데이터 유지
                "message": "Steam 회원가입 완료",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user_id": user.id,
                "redirect_url": "/"
            }, status=status.HTTP_201_CREATED)



class MyPageAPIView(APIView):
    """사용자 정보 조회, 수정, 삭제 API"""
    def get_permissions(self):
        """요청 방식(GET, PUT, DELETE)에 따라 다른 권한을 적용"""
        if self.request.method == "GET":
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def get_user(self, pk):
        return get_object_or_404(User, pk=pk)
        
    def get(self, request, pk):
        """사용자 정보 조회"""
        user = self.get_user(pk)
        serializer = UserUpdateSerializer(user)
        return Response(serializer.data)
    
    def put(self, request,pk):
        """사용자 정보 수정"""
        if pk != request.user.pk:
            return Response({"error": "You do not have permission to this page"},status=status.HTTP_403_FORBIDDEN)
        user = self.get_user(request.user.pk)
        serializer = UserUpdateSerializer(user, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)
    
    def delete(self, request, pk):
        """사용자 탈퇴 및 정보 삭제"""
        if pk != request.user.pk:
            return Response({"error": "You do not have permission to delete this user"},status=status.HTTP_403_FORBIDDEN)
        
        user = self.get_user(request.user.pk)
        user.delete()
        return Response({"message":"withdrawal"},status=status.HTTP_204_NO_CONTENT)
    

class LogoutAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")

            # refresh_token이 없을 때
            if not refresh_token:
                return Response({"error": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)

            token = RefreshToken(refresh_token)
            token.blacklist()

            return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

        except TokenError:
            return Response({"error": "Invalid or expired token."}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            # 기타 예외 발생 시
            return Response({"error": f"Token processing error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)