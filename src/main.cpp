#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


#include "glsl.h"
#include "GLinclude.h"
#include "sammon.h"
#include "vao.h"
#include "mouse.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define DATASETS "dataset/creditcard.dat"

Datasets datasets;


int width = 800, height = 600;
void reshape(GLFWwindow *window, int w, int h){
    width = w;  height = h;
}

void dataset_ver(GLFWwindow* window){
    datasets = Datasets(DATASETS);

    int datasets_size = datasets.get_size();
    int feature_size = datasets.get_feature_size();

    /*
    for(int i = 0; i < datasets_size; i++){
        Datasets::Data data = datasets.get_data(i);

        for(int j = 0; j < feature_size; j++){
            std::cout << data.feature[j] << " ";
        }

        std::cout << std::endl;
    }
    //*/

    Eigen::MatrixXd dij(datasets_size, datasets_size);

    for(int i = 0; i < datasets_size; i++){
        for(int j = 0; j < datasets_size; j++){
            Datasets::Data d1 = datasets.get_data(i), d2 = datasets.get_data(j);
            double dis = 0;
            for(int k = 0; k < feature_size; k++){
                dis += (d1.feature[k] - d2.feature[k]) * (d1.feature[k] - d2.feature[k]);
            }
            dis = sqrt(dis);
            dij(i, j) = dis;
        }
    }


    std::ofstream debug("debug.txt");
    for(int i = 0; i < datasets_size; i++){
        for(int j = 0; j < datasets_size; j++){
            debug << dij(i, j) << "\t";
        }
        debug << std::endl;
    }

    // Eigen::MatrixXd down_dim_data(datasets_size, 3);
    Datasets down_dim = Datasets(datasets_size, 3);
    pca(datasets, down_dim);

    Sammon sammon(down_dim, 3, 500000, 0.8, 0.9, 1e-6);

    int pointCount = down_dim.get_size();
    int dim = down_dim.get_feature_size();  // should be 3
    if(dim != 3){
        std::cerr << "Error: PCA output dimension must be 3, got " << dim << std::endl;
        return;
    }

    // 2. Gather vertex data
    std::vector<float> vertices;
    vertices.reserve(pointCount * 3);
    for(int i = 0; i < pointCount; ++i){
        auto data = down_dim.get_data(i);
        for(double v : data.feature){
            vertices.push_back(static_cast<float>(v));
        }
    }

    // 4. Prep shader and buffers
    Shader shader("shader/shader.vs", "shader/shader.fs");
    VertexArray va;
    // 在 main() 渲染循环之前，只做一次：
    std::vector<float> interleaved;
    interleaved.reserve(pointCount * 6);
    for(int i = 0; i < pointCount; ++i){
        auto d = down_dim.get_data(i);
        // 位置
        interleaved.push_back((float) d.feature[0]);
        interleaved.push_back((float) d.feature[1]);
        interleaved.push_back((float) d.feature[2]);
        // 颜色： target == 0 -> 蓝, target == 1 -> 红
        if(d.target == 0){
            interleaved.push_back(0.0f);
            interleaved.push_back(0.0f);
            interleaved.push_back(1.0f);
        }
        else{
            interleaved.push_back(1.0f);
            interleaved.push_back(0.0f);
            interleaved.push_back(0.0f);
        }
    }
    va.bind();
    va.setVertexBuffer(interleaved, GL_DYNAMIC_DRAW);
    va.setAttribPointer(0, 3, GL_FLOAT, 6 * sizeof(float), (void *) 0);
    va.setAttribPointer(1, 3, GL_FLOAT, 6 * sizeof(float), (void *) (3 * sizeof(float)));
    va.unbind();



    // 5. Setup camera
    int width = 800, height = 600;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), float(width) / height, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0.0f), glm::vec3(0, 1, 0));
    glm::mat4 model = glm::mat4(1.0f);

    // 6. Render loop
    while(!glfwWindowShouldClose(window)){
        sammon.train();
        sammon.get_new_data(down_dim);
        // 在 Gather vertex data 之后，建立一个 interleaved buffer：
            // 每个点： x, y, z,   r, g, b
        std::vector<float> interleaved;
        interleaved.reserve(pointCount * 6);
        for(int i = 0; i < pointCount; ++i){
            auto d = down_dim.get_data(i);
            // 位置
            interleaved.push_back((float) d.feature[0]);
            interleaved.push_back((float) d.feature[1]);
            interleaved.push_back((float) d.feature[2]);
            // 颜色： target == 0 -> 蓝, target == 1 -> 红
            if(d.target == 0){
                interleaved.push_back(0.0f);
                interleaved.push_back(0.0f);
                interleaved.push_back(1.0f);
            }
            else{
                interleaved.push_back(1.0f);
                interleaved.push_back(0.0f);
                interleaved.push_back(0.0f);
            }
        }
        va.updateVertexBuffer(interleaved);



        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        float camX = radius * cos(pitch) * sin(yaw);
        float camY = radius * sin(pitch);
        float camZ = radius * cos(pitch) * cos(yaw);
        glm::mat4 view = glm::lookAt(
            glm::vec3(camX, camY, camZ), // camera position
            glm::vec3(0.0f, 0.0f, 0.0f),   // look at origin
            glm::vec3(0.0f, 1.0f, 0.0f)    // up vector
        );
        glm::mat4 mvp = proj * view * model;

        shader.set_mat4("uMVP", mvp);
        shader.set_vec3("uColor", glm::vec3(1.0f, 0.8f, 0.2f));


        va.bind();
        // Draw lines connecting points sequentially
        // glDrawArrays(GL_LINE_STRIP, 0, pointCount);
        // Draw points
        glDrawArrays(GL_POINTS, 0, pointCount);
        va.unbind();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

}

void eigen_ver(){

}



int main(int argc, char *argv[]){
    glutInit(&argc, argv);
    if(!glfwInit()){
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(width, height, "Hw1", nullptr, nullptr);
    if(!window){
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowSizeCallback(window, reshape);



    if(!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)){
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glfwSetWindowSizeCallback(window, reshape);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);


    reshape(window, width, height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::cout << "GLFW version: " << glfwGetVersionString() << std::endl;
    const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *version = glGetString(GL_VERSION);
    std::cout << "GLSL version: " << glslVersion << std::endl;
    std::cout << "Renderer: " << renderer << std::endl;
    std::cout << "OpenGL version supported: " << version << std::endl;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 400");

    dataset_ver(window);


    glfwTerminate();
    return 0;
}
