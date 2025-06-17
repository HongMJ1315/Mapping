#version 330 core
in vec3 vColor;            // 从 vertex 传进来
out vec4 FragColor;

void main(){
    FragColor = vec4(vColor, 1.0);
}
